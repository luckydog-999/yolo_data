# -*- coding: utf-8 -*-
from tqdm import tqdm
import shutil
import random
import os
import argparse
from collections import Counter
import yaml
import json
import cv2  # 新增
import albumentations as A  # 新增
import numpy as np  # 新增

# =================================================================================
# 1. 数据增强配置区域
# =================================================================================
# 定义你的增强管道 (可以自由组合和修改)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(p=0.2),
])

# 为每张原始图片生成多少张增强后的图片
AUGMENTATIONS_PER_IMAGE = 3
# =================================================================================


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_label_json(json_dir, save_dir, classes):
    # 此函数现在仅用于调试或单独转换，主流程已被新函数替代
    json_paths = os.listdir(json_dir)
    classes = classes.split(',')
    mkdir(save_dir)

    for json_path in tqdm(json_paths):
        path = os.path.join(json_dir, json_path)
        with open(path, 'r') as load_f:
            json_dict = json.load(load_f)
        h, w = json_dict['imageHeight'], json_dict['imageWidth']

        txt_path = os.path.join(save_dir, json_path.replace('json', 'txt'))
        txt_file = open(txt_path, 'w')

        for shape_dict in json_dict['shapes']:
            label = shape_dict['label']
            label_index = classes.index(label)
            points = shape_dict['points']

            points_nor_list = []
            for point in points:
                points_nor_list.append(point[0] / w)
                points_nor_list.append(point[1] / h)

            points_nor_list = list(map(lambda x: str(x), points_nor_list))
            points_nor_str = ' '.join(points_nor_list)

            label_str = str(label_index) + ' ' + points_nor_str + '\n'
            txt_file.writelines(label_str)


def get_classes(json_dir):
    names = []
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for shape in data['shapes']:
                name = shape['label']
                names.append(name)
    result = Counter(names)
    return result


def split_dataset(all_images_dir, all_labels_dir, save_dir, json_dir_for_yaml):
    # 重命名 main 函数为 split_dataset，功能不变
    mkdir(save_dir)
    images_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'labels')
    img_train_path = os.path.join(images_dir, 'train')
    img_val_path = os.path.join(images_dir, 'val')
    label_train_path = os.path.join(labels_dir, 'train')
    label_val_path = os.path.join(labels_dir, 'val')
    mkdir(images_dir); mkdir(labels_dir); mkdir(img_train_path)
    mkdir(img_val_path); mkdir(label_train_path); mkdir(label_val_path)

    train_percent = 0.90
    val_percent = 0.10

    total_txt = os.listdir(all_labels_dir)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)

    num_train = int(num_txt * train_percent)
    train = random.sample(list_all_txt, num_train)
    val = [i for i in list_all_txt if not i in train]

    print("划分数据集: 训练集数目：{}, 验证集数目：{}".format(len(train), len(val)))
    for i in tqdm(list_all_txt):
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

    obj_classes = get_classes(json_dir_for_yaml)
    classes = list(obj_classes.keys())
    classes_txt = {i: classes[i] for i in range(len(classes))}
    data = {
        'path': os.path.abspath(save_dir),
        'train': "images/train",
        'val': "images/val",
        'names': classes_txt,
        'nc': len(classes)
    }
    with open(os.path.join(save_dir, 'segment.yaml'), 'w', encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)
    print("标签：", dict(obj_classes))
    print(f"segment.yaml 文件已在 {save_dir} 目录中生成。")

# =================================================================================
# 2. 核心数据增强函数
# =================================================================================
def augment_data(image_dir, json_dir, all_images_save_dir, all_labels_save_dir, classes_str):
    """
    对数据集进行增强，并将原始数据和增强数据全部转换为YOLO格式。
    """
    classes = classes_str.split(',')
    class_to_id = {name: i for i, name in enumerate(classes)}

    mkdir(all_images_save_dir)
    mkdir(all_labels_save_dir)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]

    for image_name in tqdm(image_files, desc="增强并转换数据中"):
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_dir, image_name)
        json_path = os.path.join(json_dir, base_name + '.json')

        if not os.path.exists(json_path):
            print(f"警告：找不到对应的JSON文件 {json_path}，跳过 {image_name}")
            continue

        # 读取原始图像和JSON
        image = cv2.imread(image_path)
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
                # 使用 OpenCV 填充多边形, 注意类别ID需要+1，因为0通常是背景
                cv2.fillPoly(mask, [points], color=(class_id + 1))

        # --- a. 处理并保存原始数据 ---
        # 复制原始图片
        shutil.copyfile(image_path, os.path.join(all_images_save_dir, image_name))
        # 将原始掩码转换为YOLO txt格式并保存
        yolo_txt_path = os.path.join(all_labels_save_dir, base_name + '.txt')
        mask_to_yolo_txt(mask, w, h, class_to_id, yolo_txt_path)

        # --- b. 进行数据增强 ---
        for i in range(AUGMENTATIONS_PER_IMAGE):
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']

            # 定义新文件名
            new_base_name = f"{base_name}_aug_{i}"
            output_image_path = os.path.join(all_images_save_dir, new_base_name + ".png")
            output_label_path = os.path.join(all_labels_save_dir, new_base_name + ".txt")

            # 保存增强后的图像
            cv2.imwrite(output_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            # 将增强后的掩码转换为YOLO格式并保存
            mask_to_yolo_txt(aug_mask, w, h, class_to_id, output_label_path)


def mask_to_yolo_txt(mask, w, h, class_to_id, save_path):
    """
    辅助函数：将numpy掩码转换为YOLO分割格式的txt文件。
    """
    yolo_lines = []
    # 掩码中的值是 class_id + 1
    unique_ids = np.unique(mask)

    for seg_val in unique_ids:
        if seg_val == 0:  # 跳过背景
            continue
        
        class_id = seg_val - 1 # 获取真实的类别ID
        
        binary_mask = np.where(mask == seg_val, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.shape[0] > 2:
                normalized_contour = contour.astype(np.float32).reshape(-1, 2)
                normalized_contour[:, 0] /= w
                normalized_contour[:, 1] /= h
                
                points_str = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in normalized_contour])
                yolo_lines.append(f"{class_id} {points_str}")

    with open(save_path, 'w') as f:
        f.write("\n".join(yolo_lines))
# =================================================================================


if __name__ == "__main__":
    classes_list = 'passive'  # 您的类名，多个类用逗号隔开，例如: 'cat,dog,person'

    parser = argparse.ArgumentParser(description='Augment, convert and split dataset for YOLO segmentation')
    parser.add_argument('--image-dir', type=str, default='./images', help='原始图片地址')
    parser.add_argument('--json-dir', type=str, default='./labels', help='原始json地址')
    parser.add_argument('--save-dir', default='datasets/segment/yolo_dataset', type=str, help='保存最终划分好的数据集地址')
    parser.add_argument('--classes', type=str, default=classes_list, help='classes, comma separated')
    args = parser.parse_args()

    # 定义中间文件夹，用于存放所有（原始+增强）的数据
    ALL_IMAGES_DIR = os.path.join('datasets/segment', 'all_images')
    ALL_LABELS_DIR = os.path.join('datasets/segment', 'all_labels')

    # --- 3. 修改后的新执行流程 ---

    # 第1步：进行数据增强，并将所有结果（原始+增强）统一转换为YOLO格式
    print("步骤 1/2: 开始数据增强与格式转换...")
    augment_data(args.image_dir, args.json_dir, ALL_IMAGES_DIR, ALL_LABELS_DIR, args.classes)
    print("增强与转换完成！")

    # 第2步：将生成的所有数据进行训练集和验证集的划分，并生成yaml文件
    print("\n步骤 2/2: 开始划分数据集...")
    split_dataset(ALL_IMAGES_DIR, ALL_LABELS_DIR, args.save_dir, args.json_dir)
    print("数据集划分完成！")
    
    # 清理中间文件夹（可选）
    # print("\n正在清理临时文件...")
    # shutil.rmtree(ALL_IMAGES_DIR)
    # shutil.rmtree(ALL_LABELS_DIR)
    # print("清理完成。")