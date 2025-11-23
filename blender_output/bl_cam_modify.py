import bpy

# ================= 配置区域 =================
# 1. 你的 ZED 矩阵数据
FX = 520.9471435546875000
FY = 520.9471435546875000
CX = 656.6704101562500000
CY = 356.9190979003906250

# 2. 你的图像分辨率 (必须确认！根据 CX 推测是 1280x720)
# 如果你用的是 2K 或 1080p，请务必修改这里
IMG_WIDTH = 1280
IMG_HEIGHT = 720
# ===========================================

def apply_intrinsics(cam_obj):
    """将内参矩阵应用到 Blender 摄像机"""
    cam = cam_obj.data
    
    # 1. 设置分辨率
    scene = bpy.context.scene
    scene.render.resolution_x = IMG_WIDTH
    scene.render.resolution_y = IMG_HEIGHT
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    
    # 2. 设置传感器适配模式 (重要！)
    # Blender 默认传感器宽度是 36mm，我们需要基于宽度来计算
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = 36.0  # 保持默认 36mm 全画幅基准即可
    
    # 3. 计算并设置焦距 (Focal Length)
    # 公式: F_mm = F_pixel * Sensor_width_mm / Image_width_pixel
    focal_length_mm = FX * cam.sensor_width / IMG_WIDTH
    cam.lens = focal_length_mm
    
    # 4. 计算并设置光心偏移 (Shift)
    # Blender 的 Shift 是归一化的，且原点在中心
    # 公式: Shift_x = -(cx - w/2) / w
    # 公式: Shift_y = (cy - h/2) / w  (注意 Blender Y轴向上，像素坐标Y轴向下)
    
    # 这里的计算取决于 Sensor Fit，如果是 Horizontal，分母通常是 Width
    shift_x = -(CX - IMG_WIDTH / 2.0) / IMG_WIDTH
    shift_y = (CY - IMG_HEIGHT / 2.0) / IMG_WIDTH * (IMG_WIDTH / IMG_HEIGHT) # 修正比例
    
    # 简化版 Shift 计算 (适用于 Horizontal Fit)
    cam.shift_x = -(CX - (IMG_WIDTH / 2.0)) / max(IMG_WIDTH, IMG_HEIGHT)
    cam.shift_y = (CY - (IMG_HEIGHT / 2.0)) / max(IMG_WIDTH, IMG_HEIGHT)

    print(f"✅ 相机校准完成！")
    print(f"   - 等效焦距: {focal_length_mm:.4f} mm")
    print(f"   - Shift X: {cam.shift_x:.4f}")
    print(f"   - Shift Y: {cam.shift_y:.4f}")

# 执行
cam_obj = bpy.context.scene.camera
if cam_obj:
    apply_intrinsics(cam_obj)
else:
    print("❌ 场景中没有摄像机！")