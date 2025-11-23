import bpy
import os
import random
import math
from mathutils import Euler

# ================== ⚙️ 配置区域 (请修改这里) ==================
# 1. 输出的总根目录 (请改为你电脑上的实际路径)
# 注意：脚本会自动在此目录下创建 "raws" 和 "masks" 文件夹
OUTPUT_ROOT = r"D:\Dataset\blender_output" 

# 2. 生成多少张图片？
TOTAL_IMAGES = 20

# 3. 你的物体名字 (在右侧大纲视图里看)
OBJ_NAME = "insert"

# 4. 物体的 Pass Index (必须与右侧 Object Properties -> Relations -> Pass Index 一致)
OBJ_INDEX = 1
# ============================================================

def setup_directories():
    """自动创建 raws 和 masks 文件夹"""
    raws_dir = os.path.join(OUTPUT_ROOT, "raws")
    masks_dir = os.path.join(OUTPUT_ROOT, "masks")
    
    if not os.path.exists(raws_dir):
        os.makedirs(raws_dir)
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
        
    return raws_dir, masks_dir

def setup_compositor_nodes(raws_dir, masks_dir):
    """配置合成器节点，将RGB和Mask分开输出"""
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    
    # 1. 清空现有节点
    for node in tree.nodes:
        tree.nodes.remove(node)
        
    # 2. 创建输入节点 (Render Layers)
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = (-300, 0)
    
    # 3. 创建 ID Mask 节点 (提取物体轮廓)
    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.index = OBJ_INDEX
    id_mask.use_antialiasing = True # 开启抗锯齿，边缘更平滑
    id_mask.location = (0, 100)
    
    # 4. 创建输出节点 - RGB 图片 (存入 raws)
    file_output_raw = tree.nodes.new('CompositorNodeOutputFile')
    file_output_raw.base_path = raws_dir
    file_output_raw.file_slots.clear()
    file_output_raw.file_slots.new("image_") # 文件名前缀，例如 image_0001.png
    file_output_raw.location = (300, 200)
    
    # 5. 创建输出节点 - Mask 图片 (存入 masks)
    file_output_mask = tree.nodes.new('CompositorNodeOutputFile')
    file_output_mask.base_path = masks_dir
    file_output_mask.file_slots.clear()
    file_output_mask.file_slots.new("mask_") # 文件名前缀，例如 mask_0001.png
    file_output_mask.location = (300, -100)
    
    # 6. 连接节点
    # 连 RGB
    links.new(render_layers.outputs['Image'], file_output_raw.inputs[0])
    
    # 连 Mask (Object Index -> ID Mask -> Output)
    links.new(render_layers.outputs['IndexOB'], id_mask.inputs[0])
    links.new(id_mask.outputs['Alpha'], file_output_mask.inputs[0])

def randomize_object(obj):
    """随机化物体的位置和旋转"""
    # 随机旋转
    obj.rotation_euler = Euler((
        random.uniform(0, math.pi * 2),
        random.uniform(0, math.pi * 2),
        random.uniform(0, math.pi * 2)
    ), 'XYZ')
    
    # 随机位置 (根据你的相机视野微调这些范围)
    obj.location.x = random.uniform(-0.15, 0.15)
    obj.location.y = random.uniform(-0.15, 0.15)
    obj.location.z = random.uniform(-0.05, 0.05)

def main():
    print("🚀 开始生成...")
    
    # 1. 准备路径
    raws_dir, masks_dir = setup_directories()
    print(f"图片将保存至: {raws_dir}")
    print(f"掩码将保存至: {masks_dir}")
    
    # 2. 获取物体
    obj = bpy.data.objects.get(OBJ_NAME)
    if not obj:
        print(f"❌ 错误: 找不到物体 '{OBJ_NAME}'，请检查名字！")
        return
        
    # 3. 确保 Object Index 正确
    obj.pass_index = OBJ_INDEX
    
    # 4. 配置合成器输出
    setup_compositor_nodes(raws_dir, masks_dir)
    
    # 5. 循环生成
    for i in range(TOTAL_IMAGES):
        # 随机化
        randomize_object(obj)
        
        # 更新场景
        bpy.context.view_layer.update()
        
        # 设置帧数 (这决定了文件名的后缀，如 0001, 0002)
        bpy.context.scene.frame_set(i + 1)
        
        # 渲染 (File Output 节点会自动保存，不需要 write_still=True)
        print(f"正在渲染第 {i+1}/{TOTAL_IMAGES} 张...")
        bpy.ops.render.render(write_still=False)
        
    print("✅ 所有图片生成完毕！")

if __name__ == "__main__":
    main()