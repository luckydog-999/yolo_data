# -*- coding: utf-8 -*-
from tqdm import tqdm
import shutil
import random
import os
import argparse
import yaml
import cv2
import albumentations as A
import numpy as np

# =================================================================================
# 1. 数据增强配置区域
# =================================================================================
# 定义增强管道
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(p=0.2),
])

# 增强倍数 (每张原图生成多少张增强图)
AUGMENTATIONS_PER_IMAGE = 3
# =================================================================================


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def yolo_txt_to_mask(txt_path, height, width):
    """
    将 YOLO 格式的 TXT 标签转换为掩码图像，以便进行数据增强。
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not os.path.exists(txt_path):
        return mask

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        # class_id
        class_id = int(parts[0])
        
        # 坐标点 (归一化 -> 像素坐标)
        coords = [float(x) for x in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i+1] * height)
            points.append([x, y])
        
        if len(points) > 0:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # 在掩码上绘制填充多边形
            # 颜色值 = class_id + 1 (为了区分背景0)
            cv2.fillPoly(mask, [pts], color=(class_id + 1))
            
    return mask

def mask_to_yolo_txt(mask, w, h, save_path):
    """
    将增强后的掩码转换回 YOLO TXT 格式。
    """
    yolo_lines = []
    unique_ids = np.unique(mask)

    for seg_val in unique_ids:
        if seg_val == 0: 
            continue
        
        # 还原真实的 class_id
        class_id = seg_val - 1 
        
        # 提取该类别的二值掩码
        binary_mask = np.where(mask == seg_val, 255, 0).astype(np.uint8)
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 过滤过小的轮廓
            if contour.shape[0] > 3:
                # 归一化坐标
                normalized_contour = contour.astype(np.float32).reshape(-1, 2)
                normalized_contour[:, 0] /= w
                normalized_contour[:, 1] /= h
                
                # 限制坐标在0-1之间
                np.clip(normalized_contour, 0, 1, out=normalized_contour)
                
                # 格式化坐标字符串
                points_str = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in normalized_contour])
                yolo_lines.append(f"{class_id} {points_str}")

    # 保存 TXT
    with open(save_path, 'w') as f:
        if yolo_lines:
            f.write("\n".join(yolo_lines))
        else:
            # 如果增强后物体消失（例如移出了画面），生成空文件
            pass 

def augment_data(image_dir, label_dir, all_images_save_dir, all_labels_save_dir):
    """
    读取图片和TXT标签 -> 转掩码 -> 增强 -> 转回TXT -> 保存
    """
    mkdir(all_images_save_dir)
    mkdir(all_labels_save_dir)

    # 支持常见的图片格式
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for image_name in tqdm(image_files, desc="数据增强处理中"):
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_dir, image_name)
        
        # 寻找对应的 txt 文件
        txt_name = base_name + '.txt'
        txt_path = os.path.join(label_dir, txt_name)

        if not os.path.exists(txt_path):
            print(f"警告：找不到对应的标签文件 {txt_path}，跳过 {image_name}")
            continue

        # 1. 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 2. 读取 TXT 并转换为掩码
        mask = yolo_txt_to_mask(txt_path, h, w)

        # 3. 保存原始数据 (复制图片和标签)
        shutil.copyfile(image_path, os.path.join(all_images_save_dir, image_name))
        shutil.copyfile(txt_path, os.path.join(all_labels_save_dir, txt_name))

        # 4. 生成增强数据
        for i in range(AUGMENTATIONS_PER_IMAGE):
            # 应用增强
            try:
                augmented = transform(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']

                # 定义新文件名
                new_base_name = f"{base_name}_aug_{i}"
                output_image_path = os.path.join(all_images_save_dir, new_base_name + ".png") # 统一保存为png防止压缩损失
                output_label_path = os.path.join(all_labels_save_dir, new_base_name + ".txt")

                # 保存图片 (转回 BGR)
                cv2.imwrite(output_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                # 保存标签 (掩码 -> TXT)
                mask_to_yolo_txt(aug_mask, w, h, output_label_path)
            
            except Exception as e:
                print(f"增强 {image_name} 时出错: {e}")

def split_dataset(all_images_dir, all_labels_dir, classes_str):
    """
    划分训练集和验证集，并生成 segment.yaml
    """
    # 定义根目录下的目标路径
    root_dir = '.'  # 当前根目录
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')
    
    img_train_path = os.path.join(images_dir, 'train')
    img_val_path = os.path.join(images_dir, 'val')
    label_train_path = os.path.join(labels_dir, 'train')
    label_val_path = os.path.join(labels_dir, 'val')

    # 创建目录
    mkdir(images_dir); mkdir(labels_dir)
    mkdir(img_train_path); mkdir(img_val_path)
    mkdir(label_train_path); mkdir(label_val_path)

    # 划分比例
    train_percent = 0.90
    
    # 获取所有标签文件
    total_txt = [f for f in os.listdir(all_labels_dir) if f.endswith('.txt')]
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)

    num_train = int(num_txt * train_percent)
    train = random.sample(list_all_txt, num_train)
    val = [i for i in list_all_txt if not i in train]

    print(f"数据集划分: 训练集 {len(train)} 张, 验证集 {len(val)} 张")

    for i in tqdm(list_all_txt, desc="分配文件到 dataset 目录"):
        txt_filename = total_txt[i]
        base_name = os.path.splitext(txt_filename)[0]
        
        # 寻找对应的图片 (可能是 .png 或 .jpg)
        srcLabel = os.path.join(all_labels_dir, txt_filename)
        
        # 尝试寻找对应的图片文件
        srcImage = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            temp_path = os.path.join(all_images_dir, base_name + ext)
            if os.path.exists(temp_path):
                srcImage = temp_path
                break
        
        if srcImage is None:
            # print(f"警告: 找不到标签 {txt_filename} 对应的图片")
            continue

        img_filename = os.path.basename(srcImage)

        if i in train:
            shutil.copyfile(srcImage, os.path.join(img_train_path, img_filename))
            shutil.copyfile(srcLabel, os.path.join(label_train_path, txt_filename))
        else:
            shutil.copyfile(srcImage, os.path.join(img_val_path, img_filename))
            shutil.copyfile(srcLabel, os.path.join(label_val_path, txt_filename))

    # 生成 segment.yaml
    classes_list = classes_str.split(',')
    # 构造 names 字典: {0: 'cat', 1: 'dog'}
    names_dict = {i: name for i, name in enumerate(classes_list)}
    
    data = {
        'path': os.path.abspath(root_dir),
        'train': "images/train",
        'val': "images/val",
        'names': names_dict,
        'nc': len(classes_list)
    }
    
    yaml_path = os.path.join(root_dir, 'segment.yaml')
    with open(yaml_path, 'w', encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)
        
    print(f"配置文件生成完毕: {yaml_path}")
    print(f"类别信息: {names_dict}")


if __name__ == "__main__":
    # 默认类别名称
    default_classes = 'passive' 

    parser = argparse.ArgumentParser(description='YOLO TXT Dataset Augmentation and Split')
    
    # 输入参数：原始图片和原始txt标签所在的文件夹
    parser.add_argument('--image-dir', type=str, default='./raw_images', help='存放原始图片的文件夹路径')
    parser.add_argument('--label-dir', type=str, default='./raw_labels', help='存放原始TXT标签的文件夹路径')
    parser.add_argument('--classes', type=str, default=default_classes, help='类别名称，用逗号分隔 (例如: cat,dog)')
    
    args = parser.parse_args()

    # 临时文件夹
    ALL_IMAGES_DIR = './temp_all_images'
    ALL_LABELS_DIR = './temp_all_labels'

    # 检查输入目录
    if not os.path.exists(args.image_dir) or not os.path.exists(args.label_dir):
        print("❌ 错误：输入目录不存在！")
        print(f"请检查 --image-dir ({args.image_dir}) 和 --label-dir ({args.label_dir})")
        exit()

    # 1. 增强
    print("\n>>> 步骤 1/2: 数据增强...")
    augment_data(args.image_dir, args.label_dir, ALL_IMAGES_DIR, ALL_LABELS_DIR)
    print("✅ 增强完成")

    # 2. 划分
    print("\n>>> 步骤 2/2: 划分数据集...")
    split_dataset(ALL_IMAGES_DIR, ALL_LABELS_DIR, args.classes)
    print("✅ 划分完成")
    
    # 3. 清理
    print("\n正在清理临时文件...")
    shutil.rmtree(ALL_IMAGES_DIR)
    shutil.rmtree(ALL_LABELS_DIR)
    print("🎉 全部搞定！数据集已保存在 ./images 和 ./labels，配置文件为 segment.yaml")