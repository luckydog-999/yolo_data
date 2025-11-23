# -*- coding: utf-8 -*-
from tqdm import tqdm
import shutil
import random
import os
import argparse
from collections import Counter
import yaml
import json
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

# 增强倍数
AUGMENTATIONS_PER_IMAGE = 3
# =================================================================================


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_classes(json_dir):
    names = []
    # 注意：这里只统计原始JSON中的类别
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for shape in data['shapes']:
                name = shape['label']
                names.append(name)
    result = Counter(names)
    return result


def split_dataset(all_images_dir, all_labels_dir, json_dir_for_yaml):
    """
    修改后的划分函数：
    将数据直接划分为根目录下的 ./images/train, ./images/val 和 ./labels/train, ./labels/val
    """
    # 定义根目录下的目标路径
    root_dir = '.'  # 当前根目录
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')
    
    img_train_path = os.path.join(images_dir, 'train')
    img_val_path = os.path.join(images_dir, 'val')
    label_train_path = os.path.join(labels_dir, 'train')
    label_val_path = os.path.join(labels_dir, 'val')

    # 创建目录结构
    mkdir(images_dir)
    mkdir(labels_dir)
    mkdir(img_train_path)
    mkdir(img_val_path)
    mkdir(label_train_path)
    mkdir(label_val_path)

    train_percent = 0.90
    
    # 获取所有增强后生成的txt标签文件列表
    total_txt = os.listdir(all_labels_dir)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)

    num_train = int(num_txt * train_percent)
    train = random.sample(list_all_txt, num_train)
    val = [i for i in list_all_txt if not i in train]

    print(f"目标路径: ./images 和 ./labels")
    print(f"划分数据集: 训练集数目：{len(train)}, 验证集数目：{len(val)}")

    for i in tqdm(list_all_txt, desc="划分并移动文件"):
        name = total_txt[i][:-4]
        srcImage = os.path.join(all_images_dir, name + '.png')
        srcLabel = os.path.join(all_labels_dir, name + '.txt')

        if i in train:
            dst_train_Image = os.path.join(img_train_path, name + '.png')
            dst_train_Label = os.path.join(label_train_path, name + '.txt')
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
        elif i in val:
            dst_val_Image = os.path.join(img_val_path, name + '.png')
            dst_val_Label = os.path.join(label_val_path, name + '.txt')
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)

    # 生成 yaml 文件
    obj_classes = get_classes(json_dir_for_yaml)
    classes = list(obj_classes.keys())
    classes_txt = {i: classes[i] for i in range(len(classes))}
    
    data = {
        'path': os.path.abspath(root_dir), # 使用绝对路径避免错误
        'train': "images/train",
        'val': "images/val",
        'names': classes_txt,
        'nc': len(classes)
    }
    
    yaml_path = os.path.join(root_dir, 'segment.yaml')
    with open(yaml_path, 'w', encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)
        
    print("标签统计：", dict(obj_classes))
    print(f"配置文件已生成：{yaml_path}")


# =================================================================================
# 2. 核心数据增强函数 (保持不变，负责生成中间数据)
# =================================================================================
def augment_data(image_dir, json_dir, all_images_save_dir, all_labels_save_dir, classes_str):
    classes = classes_str.split(',')
    class_to_id = {name: i for i, name in enumerate(classes)}

    mkdir(all_images_save_dir)
    mkdir(all_labels_save_dir)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png') or f.lower().endswith('.jpg')]

    for image_name in tqdm(image_files, desc="增强并转换数据中"):
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_dir, image_name)
        json_path = os.path.join(json_dir, base_name + '.json')

        if not os.path.exists(json_path):
            print(f"警告：找不到对应的JSON文件 {json_path}，跳过 {image_name}")
            continue

        # 读取原始图像和JSON
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(json_path, 'r') as f:
            data = json.load(f)

        h, w = data['imageHeight'], data['imageWidth']
        mask = np.zeros((h, w), dtype=np.uint8)

        # 从JSON创建掩码
        for shape in data['shapes']:
            label = shape['label']
            if label in class_to_id:
                class_id = class_to_id[label]
                points = np.array(shape['points'], dtype=np.int32)
                # 类别ID+1，0作为背景
                cv2.fillPoly(mask, [points], color=(class_id + 1))

        # --- a. 处理并保存原始数据 ---
        shutil.copyfile(image_path, os.path.join(all_images_save_dir, image_name))
        yolo_txt_path = os.path.join(all_labels_save_dir, base_name + '.txt')
        mask_to_yolo_txt(mask, w, h, class_to_id, yolo_txt_path)

        # --- b. 进行数据增强 ---
        for i in range(AUGMENTATIONS_PER_IMAGE):
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']

            new_base_name = f"{base_name}_aug_{i}"
            output_image_path = os.path.join(all_images_save_dir, new_base_name + ".png")
            output_label_path = os.path.join(all_labels_save_dir, new_base_name + ".txt")

            cv2.imwrite(output_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            mask_to_yolo_txt(aug_mask, w, h, class_to_id, output_label_path)


def mask_to_yolo_txt(mask, w, h, class_to_id, save_path):
    yolo_lines = []
    unique_ids = np.unique(mask)

    for seg_val in unique_ids:
        if seg_val == 0: 
            continue
        
        class_id = seg_val - 1 
        
        binary_mask = np.where(mask == seg_val, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.shape[0] > 2:
                normalized_contour = contour.astype(np.float32).reshape(-1, 2)
                normalized_contour[:, 0] /= w
                normalized_contour[:, 1] /= h
                
                # 限制坐标在0-1之间，防止增强导致越界
                np.clip(normalized_contour, 0, 1, out=normalized_contour)
                
                points_str = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in normalized_contour])
                yolo_lines.append(f"{class_id} {points_str}")

    with open(save_path, 'w') as f:
        f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    # 这里设置您的类别，如果不是 passive 请修改
    classes_list = 'passive' 

    parser = argparse.ArgumentParser(description='YOLO Segmentation Dataset Preparation')
    # 建议将原始素材放在 raw_data 或 origin 文件夹，避免和生成的 ./images 冲突
    parser.add_argument('--image-dir', type=str, default='./raw_images', help='原始图片存放文件夹')
    parser.add_argument('--json-dir', type=str, default='./raw_json', help='原始json存放文件夹')
    parser.add_argument('--classes', type=str, default=classes_list, help='类别名称')
    args = parser.parse_args()

    # 1. 设置中间临时目录 (生成完后可以删除)
    ALL_IMAGES_DIR = './temp_all_images'
    ALL_LABELS_DIR = './temp_all_labels'

    if not os.path.exists(args.image_dir) or not os.path.exists(args.json_dir):
        print(f"错误：输入目录不存在。请确保原始图片在 {args.image_dir}，原始JSON在 {args.json_dir}")
        print("或者通过命令行参数指定: python prepare.py --image-dir 你的图片目录 --json-dir 你的json目录")
        exit()

    # 2. 执行增强和转换 -> 存入临时目录
    print("步骤 1/2: 数据增强与格式转换...")
    augment_data(args.image_dir, args.json_dir, ALL_IMAGES_DIR, ALL_LABELS_DIR, args.classes)
    print("增强完成！")

    # 3. 划分数据集 -> 存入根目录 ./images 和 ./labels
    print("\n步骤 2/2: 划分数据集到根目录 ./images 和 ./labels ...")
    split_dataset(ALL_IMAGES_DIR, ALL_LABELS_DIR, args.json_dir)
    print("完成！")
    
    # 4. 清理临时文件
    print("清理临时文件...")
    shutil.rmtree(ALL_IMAGES_DIR)
    shutil.rmtree(ALL_LABELS_DIR)
    print("清理完毕。现在您可以直接使用 segment.yaml 开始训练。")