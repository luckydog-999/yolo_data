import bpy
import os

# ================= 配置区域 =================
# !!! 请在这里修改你想保存图片的文件夹路径 !!!
# Windows 例子: r"D:\Dataset\Bolt_Data"
# 注意前缀 r 很重要，或者用双斜杠 \\
OUTPUT_DIR = r"D:\My_Blender_Dataset\Test_01"

# 你的物体 Index 设的是多少？(刚才我们设的是 1)
TARGET_OBJECT_INDEX = 1
# ===========================================

def setup_compositor_nodes():
    # 1. 启用合成节点
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    
    # 2. 清空现有节点（防止重复添加）
    for node in tree.nodes:
        tree.nodes.remove(node)
        
    # 3. 创建核心节点
    # 输入：渲染层
    render_layers_node = tree.nodes.new('CompositorNodeRLayers')
    render_layers_node.location = (-300, 0)
    
    # 处理：ID Mask (用来提取你的物体)
    id_mask_node = tree.nodes.new('CompositorNodeIDMask')
    id_mask_node.index = TARGET_OBJECT_INDEX # 设置为 1
    id_mask_node.use_antialiasing = True # 边缘抗锯齿，让分割更平滑
    id_mask_node.location = (0, 100)
    
    # 输出：文件输出 (File Output)
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')
    file_output_node.base_path = OUTPUT_DIR
    file_output_node.location = (300, 0)
    
    # 4. 配置输出格式
    # 清除默认插槽，我们要自己建两个
    file_output_node.file_slots.clear()
    
    # 插槽 1: RGB 图片 (命名为 image)
    slot_image = file_output_node.file_slots.new("image")
    
    # 插槽 2: Mask 掩码 (命名为 mask)
    slot_mask = file_output_node.file_slots.new("mask")
    
    # 5. 连接节点
    links = tree.links
    
    # 连接 RGB 图片
    links.new(render_layers_node.outputs['Image'], file_output_node.inputs['image'])
    
    # 连接 Mask (从 Object Index -> ID Mask -> Output)
    # 注意：这里需要连接 Render Layers 的 'IndexOB' (Object Index)
    links.new(render_layers_node.outputs['IndexOB'], id_mask_node.inputs[0])
    links.new(id_mask_node.outputs['Alpha'], file_output_node.inputs['mask'])
    
    print("节点配置完成！")

def render_single_frame():
    # 确保路径存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 执行渲染 (write_still=True 表示渲染后保存)
    bpy.ops.render.render(write_still=True)
    print(f"渲染完成！请去 {OUTPUT_DIR} 查看结果")

# === 执行主程序 ===
setup_compositor_nodes()  # 设置节点
render_single_frame()     # 渲染一张试试