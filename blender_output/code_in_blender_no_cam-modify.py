import bpy
import os
import random
import math
import re
import sys
import subprocess
from mathutils import Euler, Vector

# ================== 📦 自动环境配置 ==================
try:
    from tqdm import tqdm
except ImportError:
    print("正在为 Blender 安装 tqdm 库，请稍候...")
    python_exe = sys.executable
    subprocess.call([python_exe, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm
# ====================================================

# ================== ⚙️ 配置区域 ==================
OUTPUT_ROOT = r"C:\Users\29746\Desktop\mesh\yolo_prepare\blender_output" 
DIR_NAME_RAW = "raws"
DIR_NAME_MASK = "masks"

BATCH_SIZE = 100
OBJ_NAME = "insert"
OBJ_INDEX = 1

# 相机随机移动的距离范围 (单位: 米)
# 脚本只会移动相机位置
CAM_DIST_MIN = 0.2
CAM_DIST_MAX = 0.85
# =================================================

# --- 静音模式 (防止 Saved 刷屏) ---
class SuppressOutput:
    def __enter__(self):
        self.stdout_fileno = sys.stdout.fileno()
        self.stderr_fileno = sys.stderr.fileno()
        self.saved_stdout = os.dup(self.stdout_fileno)
        self.saved_stderr = os.dup(self.stderr_fileno)
        self.null = os.open(os.devnull, os.O_RDWR)
        os.dup2(self.null, self.stdout_fileno)
        os.dup2(self.null, self.stderr_fileno)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout, self.stdout_fileno)
        os.dup2(self.saved_stderr, self.stderr_fileno)
        os.close(self.null)
        os.close(self.saved_stdout)
        os.close(self.saved_stderr)

def setup_directories():
    raws_dir = os.path.join(OUTPUT_ROOT, DIR_NAME_RAW)
    masks_dir = os.path.join(OUTPUT_ROOT, DIR_NAME_MASK)
    if not os.path.exists(raws_dir): os.makedirs(raws_dir)
    if not os.path.exists(masks_dir): os.makedirs(masks_dir)
    return raws_dir, masks_dir

def get_next_start_index(directory, prefix="image_"):
    if not os.path.exists(directory): return 1
    files = os.listdir(directory)
    max_num = 0
    pattern = re.compile(rf"{prefix}(\d+)\.")
    for f in files:
        match = pattern.search(f)
        if match:
            num = int(match.group(1))
            if num > max_num: max_num = num
    return max_num + 1

def setup_compositor_nodes(raws_dir, masks_dir):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for node in tree.nodes: tree.nodes.remove(node)
    
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = (-300, 0)
    
    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.index = OBJ_INDEX
    id_mask.use_antialiasing = True
    id_mask.location = (0, 100)
    
    out_raw = tree.nodes.new('CompositorNodeOutputFile')
    out_raw.base_path = raws_dir
    out_raw.file_slots.clear()
    out_raw.file_slots.new("image_")
    out_raw.location = (300, 200)
    
    out_mask = tree.nodes.new('CompositorNodeOutputFile')
    out_mask.base_path = masks_dir
    out_mask.file_slots.clear()
    out_mask.file_slots.new("mask_")
    out_mask.location = (300, -100)
    
    tree.links.new(rl.outputs['Image'], out_raw.inputs[0])
    tree.links.new(rl.outputs['IndexOB'], id_mask.inputs[0])
    tree.links.new(id_mask.outputs['Alpha'], out_mask.inputs[0])

def create_or_get_floor():
    floor_name = "Auto_Floor"
    floor = bpy.data.objects.get(floor_name)
    if not floor:
        bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, -0.05))
        floor = bpy.context.object
        floor.name = floor_name
        floor.pass_index = 0 
        mat = bpy.data.materials.new(name="Floor_Mat")
        mat.use_nodes = True
        floor.data.materials.append(mat)
        floor.is_shadow_catcher = True 
    return floor

def setup_camera_tracking(target_obj):
    cam = bpy.context.scene.camera
    if not cam: return
    # 依然保留追踪功能，确保相机看着物体
    for c in cam.constraints:
        if c.type == 'TRACK_TO': cam.constraints.remove(c)
    track = cam.constraints.new(type='TRACK_TO')
    track.target = target_obj
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

def randomize_object_landing(obj):
    obj.rotation_euler = Euler((
        random.uniform(0, math.pi * 2), random.uniform(0, math.pi * 2), random.uniform(0, math.pi * 2)
    ), 'XYZ')
    obj.location.x = random.uniform(-0.1, 0.1)
    obj.location.y = random.uniform(-0.1, 0.1)
    obj.location.z = 0
    
    bpy.context.view_layer.update()
    world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min([v.z for v in world_corners])
    obj.location.z -= min_z 
    obj.location.z += 0.0002

def randomize_camera_upper_hemisphere():
    cam = bpy.context.scene.camera
    # 这里只改变相机位置 (Location)，不改变 Lens (焦距) 或 Shift (偏移)
    dist = random.uniform(CAM_DIST_MIN, CAM_DIST_MAX)
    phi = random.uniform(0, math.radians(85)) 
    theta = random.uniform(0, math.pi * 2)
    x = dist * math.sin(phi) * math.cos(theta)
    y = dist * math.sin(phi) * math.sin(theta)
    z = dist * math.cos(phi)
    cam.location.x = x
    cam.location.y = y
    cam.location.z = z

def main():
    print("\n" + "="*50)
    print("🚀 YOLO 数据集生成器 (自由相机版)")
    print("="*50)
    
    obj = bpy.data.objects.get(OBJ_NAME)
    cam = bpy.context.scene.camera
    if not obj or not cam:
        print("❌ 错误: 场景中找不到物体或摄像机！")
        return
        
    # 初始化
    obj.pass_index = OBJ_INDEX
    create_or_get_floor()
    
    # 注意：这里删除了 apply_zed_intrinsics()
    # 现在你可以手动在 Blender 右侧面板设置任何你想要的焦距和分辨率
    
    setup_camera_tracking(obj) # 依然开启追踪，不然相机会乱看
    path_raw, path_mask = setup_directories()
    setup_compositor_nodes(path_raw, path_mask)
    
    start_idx = get_next_start_index(path_raw, "image_")
    end_idx = start_idx + BATCH_SIZE
    
    print(f"📂 输出目录: {path_raw}")
    print(f"📸 当前相机焦距: {cam.data.lens}mm (如需修改请在界面调整)")
    print(f"🎯 开始渲染 {BATCH_SIZE} 张图片...\n")
    
    with tqdm(total=BATCH_SIZE, desc="渲染进度", unit="img", ncols=100) as pbar:
        for current_frame in range(start_idx, end_idx):
            randomize_object_landing(obj)
            randomize_camera_upper_hemisphere()
            
            bpy.context.view_layer.update()
            bpy.context.scene.frame_set(current_frame)
            
            try:
                with SuppressOutput():
                    bpy.ops.render.render(write_still=False)
            except Exception as e:
                bpy.ops.render.render(write_still=False)
                
            pbar.update(1)
            
    print("\n" + "="*50)
    print("✅ 全部完成！")
    print("="*50)

if __name__ == "__main__":
    main()