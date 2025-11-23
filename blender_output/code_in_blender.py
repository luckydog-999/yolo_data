import bpy
import os
import random
import math
from mathutils import Euler

# ================== ⚙️ 配置区域 ==================
OUTPUT_ROOT = r"C:\Users\29746\Desktop\mesh\yolo_prepare\blender_output" 
TOTAL_IMAGES = 100
OBJ_NAME = "insert"
OBJ_INDEX = 1
# =================================================

def setup_directories():
    raws_dir = os.path.join(OUTPUT_ROOT, "raws")
    masks_dir = os.path.join(OUTPUT_ROOT, "masks")
    if not os.path.exists(raws_dir): os.makedirs(raws_dir)
    if not os.path.exists(masks_dir): os.makedirs(masks_dir)
    return raws_dir, masks_dir

def setup_compositor_nodes(raws_dir, masks_dir):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    
    # 清理旧节点
    for node in tree.nodes: tree.nodes.remove(node)
        
    # 创建节点
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = (-300, 0)
    
    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.index = OBJ_INDEX
    id_mask.use_antialiasing = True
    id_mask.location = (0, 100)
    
    file_output_raw = tree.nodes.new('CompositorNodeOutputFile')
    file_output_raw.base_path = raws_dir
    file_output_raw.file_slots.clear()
    file_output_raw.file_slots.new("image_")
    file_output_raw.location = (300, 200)
    
    file_output_mask = tree.nodes.new('CompositorNodeOutputFile')
    file_output_mask.base_path = masks_dir
    file_output_mask.file_slots.clear()
    file_output_mask.file_slots.new("mask_")
    file_output_mask.location = (300, -100)
    
    # 连接
    links = tree.links
    links.new(render_layers.outputs['Image'], file_output_raw.inputs[0])
    links.new(render_layers.outputs['IndexOB'], id_mask.inputs[0])
    links.new(id_mask.outputs['Alpha'], file_output_mask.inputs[0])

def setup_camera_tracking(target_obj):
    """
    让场景中的摄像机始终盯着目标物体
    """
    # 获取当前场景的摄像机
    cam = bpy.context.scene.camera
    if not cam:
        print("❌ 错误: 场景中没有摄像机！请添加一个摄像机。")
        return

    # 清除已有的约束（防止重复添加）
    for constraint in cam.constraints:
        if constraint.type == 'TRACK_TO':
            cam.constraints.remove(constraint)

    # 添加 'Track To' 约束
    track = cam.constraints.new(type='TRACK_TO')
    track.target = target_obj
    track.track_axis = 'TRACK_NEGATIVE_Z' # -Z 轴对准物体（Blender相机默认朝向）
    track.up_axis = 'UP_Y'                # Y 轴向上
    
    print(f"🎥 摄像机已锁定目标: {target_obj.name}")

def randomize_object(obj):
    # 1. 随机旋转
    obj.rotation_euler = Euler((
        random.uniform(0, math.pi * 2),
        random.uniform(0, math.pi * 2),
        random.uniform(0, math.pi * 2)
    ), 'XYZ')
    
    # 2. 随机位置 (范围可以稍微大一点了，因为摄像机会跟着转)
    # 注意：不要让物体离摄像机太近或太远导致裁剪
    obj.location.x = random.uniform(-0.2, 0.2)
    obj.location.y = random.uniform(-0.2, 0.2)
    obj.location.z = random.uniform(-0.2, 0.2)

def main():
    print("🚀 开始生成...")
    
    # 1. 准备路径
    raws_dir, masks_dir = setup_directories()
    
    # 2. 获取物体
    obj = bpy.data.objects.get(OBJ_NAME)
    if not obj:
        print(f"❌ 错误: 找不到物体 '{OBJ_NAME}'")
        return
        
    obj.pass_index = OBJ_INDEX
    
    # 3. 设置节点和摄像机追踪
    setup_compositor_nodes(raws_dir, masks_dir)
    setup_camera_tracking(obj) # <--- 新增：锁定视角
    
    # 4. 循环生成
    for i in range(TOTAL_IMAGES):
        randomize_object(obj)
        
        # 关键：更新场景矩阵，确保约束生效
        bpy.context.view_layer.update()
        
        bpy.context.scene.frame_set(i + 1)
        print(f"正在渲染第 {i+1}/{TOTAL_IMAGES} 张...")
        bpy.ops.render.render(write_still=False)
        
    print("✅ 生成完毕！")

if __name__ == "__main__":
    main()