# train_segment.py

from ultralytics import YOLO
import torch
import os

def main():
    # =================================================================================
    # --- 1. 硬件与环境检查 (最佳实践) ---
    # =================================================================================
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"✅ 检测到 {gpu_count} 个可用的GPU。")
        print(f"🚀 当前使用GPU: Device {current_device} - {gpu_name}")
    else:
        print("⚠️ 警告：未检测到可用的GPU，将使用CPU进行训练。这会非常慢！")

    # =================================================================================
    # --- 2. 模型选择 (精度与资源平衡的关键) ---
    # =================================================================================
    # 对于您当前的数据集规模，'s' (small) 模型是最佳起点。
    # 'n' (nano) 模型太小，容易学习不充分 (欠拟合)。
    # 'm' (medium) 模型参数更多，可能达到更高精度，但需要更好的GPU和更长的训练时间。
    # 策略：先用 's' 模型训练出一个强大的基线。
    model_size = 'weights/yolo11m-seg.pt'
    
    print(f"🧠 正在加载预训练模型: {model_size}")
    model = YOLO(model_size)

    # =================================================================================
    # --- 3. 终极训练参数配置 (核心区域) ---
    # =================================================================================
    # 这里的参数经过精心设计，旨在最大化精度和泛化能力
    results = model.train(
        # --- A. 核心与硬件配置 ---
        data='datasets/segment/yolo_dataset/segment.yaml',  # 确保路径指向您最终生成的YAML文件
        imgsz=640,                                 # 输入图像尺寸，640是通用最佳尺寸
        batch=8,                                   # 批量大小。8是很好的起点。如果显存溢出(CUDA out of memory)，降至4。
        device=0,                                  # 使用第一张GPU
        workers=8,                                 # 数据加载线程数，可根据CPU核心数调整 (建议为核心数的一半)

        # --- B. 训练策略与早停机制 ---
        epochs=400,                                # 给予充足的训练轮次。不用担心过长，早停机制会处理。
        patience=50,                               # ✨ 核心早停参数：更有耐心地等待。在50个轮次内，如果验证集指标没有创下新高，则停止训练。
                                                   # 这可以帮助模型越过暂时的性能平台期，找到真正的最优解。
        optimizer='AdamW',                         # 使用AdamW优化器，通常比默认优化器收敛更快，效果更稳定。
        lr0=0.001,                                 # 初始学习率 (Initial Learning Rate)。配合AdamW，0.001是更稳健的选择。
        lrf=0.01,                                  # 最终学习率因子 (Final Learning Rate Factor)。学习率将从lr0衰减到lr0*lrf。

        # --- C. 强力数据增强 (防止过拟合、提升精度的关键！) ---
        # YOLOv8内置的在线数据增强，能极大丰富数据多样性，强迫模型学习本质特征。
        hsv_h=0.015,                               # 色调(Hue)增强
        hsv_s=0.7,                                 # 饱和度(Saturation)增强
        hsv_v=0.4,                                 # 亮度(Value)增强
        degrees=15.0,                              # 随机旋转角度范围 (+/- 15度)
        translate=0.1,                             # 随机平移范围
        scale=0.6,                                 # 随机缩放范围 (+/- 60%)，让模型适应不同大小的物体
        shear=2.0,                                 # 随机错切角度
        flipud=0.5,                                # 50%概率上下翻转
        fliplr=0.5,                                # 50%概率左右翻转
        mosaic=1.0,                                # 100%概率使用Mosaic数据增强 (拼接4张图)，对检测不同尺寸物体极度有效
        mixup=0.15,                                # 15%概率使用MixUp数据增强 (混合2张图)
        copy_paste=0.15,                           # 15%概率使用Copy-Paste增强，对分割任务尤其有效！

        # --- D. 输出与日志配置 ---
        project='YOLOv8_Training_Results',         # 所有训练结果输出的根目录
        name='object_seg_s_model_final',           # 本次训练的实验名，清晰明了
        exist_ok=False,                            # 若实验名已存在，自动创建新文件夹 (e.g., final2, final3)
        verbose=True,                              # 打印详细训练日志
        save_period=10,                            # 每10个epoch额外保存一次模型快照，以防意外中断
    )
    
    # --- 4. 训练结束总结 ---
    print("\n" + "="*50)
    print("🎉 训练已根据早停机制完成！ 🎉")
    print(f"📈 最终模型和所有训练结果保存在: {os.path.abspath(results.save_dir)}")
    print(f"🏆 最佳性能模型权重位于: {os.path.abspath(os.path.join(results.save_dir, 'weights/best.pt'))}")
    print("="*50)

if __name__ == '__main__':
    main()