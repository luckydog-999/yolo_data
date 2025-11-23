import cv2
import os
import glob

# ================= ⚙️ 配置区域 =================
# 1. 你的掩码图片文件夹路径 (Blender 生成的 masks 文件夹)
MASK_DIR = r"C:\Users\29746\Desktop\mesh\yolo_prepare\blender_output\masks"

# 2. 你希望保存 txt 标签的文件夹路径 (建议新建一个 labels 文件夹)
LABEL_DIR = r"C:\Users\29746\Desktop\mesh\yolo_prepare\blender_output\labels"

# 3. YOLO 类别 ID (你的物体是第几类？通常单物体是 0)
CLASS_ID = 0
# ==============================================

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_mask_to_yolo(mask_path, output_path):
    # 1. 读取图片 (灰度模式)
    # 即使是黑白图，OpenCV 默认也会读成 3 通道，所以必须指定 0 (IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, 0)
    
    if mask is None:
        print(f"❌ 无法读取: {mask_path}")
        return

    # 获取图像尺寸
    height, width = mask.shape

    # 2. 二值化处理 (确保只有纯黑和纯白)
    # 阈值设为 127，大于 127 变 255(白)，小于变 0(黑)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. 查找轮廓 (Find Contours)
    # EXTERNAL: 只找最外层轮廓 (如果物体中间有孔，YOLO分割通常也只需要外轮廓，除非你要抠得很细)
    # SIMPLE: 简化坐标点 (比如一条直线只需要起点和终点，能大大减小 txt 大小)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 写入 txt 文件
    yolo_lines = []
    
    for contour in contours:
        # 过滤掉太小的噪点轮廓 (面积小于 100 像素的忽略)
        if cv2.contourArea(contour) < 100:
            continue

        # 展平轮廓数组 (从 [[[x,y]], [[x,y]]] 变成 [[x,y], [x,y]])
        contour = contour.flatten()
        
        # 格式化为 YOLO 分割格式: <class-id> <x1> <y1> <x2> <y2> ...
        # 坐标必须归一化 (除以宽高)
        line_content = [str(CLASS_ID)]
        
        for i in range(0, len(contour), 2):
            x = contour[i] / width
            y = contour[i+1] / height
            
            # 限制在 0-1 之间 (防止边缘溢出)
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            
            line_content.append(f"{x:.6f} {y:.6f}")
            
        yolo_lines.append(" ".join(line_content))

    # 如果找到了轮廓，保存文件
    if yolo_lines:
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_lines))
            # print(f"✅ 生成: {os.path.basename(output_path)}")
    else:
        print(f"⚠️ 警告: {os.path.basename(mask_path)} 里没找到物体轮廓！")

def main():
    print("🚀 开始转换 Mask 到 YOLO Txt...")
    mkdir(LABEL_DIR)
    
    # 获取所有 mask 图片 (支持 png, jpg)
    mask_files = glob.glob(os.path.join(MASK_DIR, "*.png")) + glob.glob(os.path.join(MASK_DIR, "*.jpg"))
    
    if not mask_files:
        print("❌ 错误: masks 文件夹里没有图片！")
        return

    count = 0
    for mask_file in mask_files:
        # 获取文件名 (不带后缀)，例如 mask_0001
        filename = os.path.splitext(os.path.basename(mask_file))[0]
        
        # ⚠️ 关键步骤：文件名匹配
        # 如果你的 mask 叫 "mask_0001.png"，但 YOLO 训练图叫 "image_0001.png"
        # 那么 txt 必须叫 "image_0001.txt" 才能对应上。
        # 这里做一个简单的替换：把 "mask" 替换成 "image"
        txt_filename = filename.replace("mask", "image") + ".txt"
        
        output_path = os.path.join(LABEL_DIR, txt_filename)
        
        convert_mask_to_yolo(mask_file, output_path)
        count += 1
        
    print(f"🎉 转换完成！共处理 {count} 张图片。")
    print(f"📂 结果保存在: {LABEL_DIR}")

if __name__ == "__main__":
    main()