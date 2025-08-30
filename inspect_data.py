#!/usr/bin/env python3
"""
检测数据batch的脚本 - 查看图片和标签
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_fashion_mnist_dataset

# FashionMNIST的类别标签
FASHION_MNIST_LABELS = [
    'T-shirt/top',     # 0
    'Trouser',         # 1 
    'Pullover',        # 2
    'Dress',           # 3
    'Coat',            # 4
    'Sandal',          # 5
    'Shirt',           # 6
    'Sneaker',         # 7
    'Bag',             # 8
    'Ankle boot'       # 9
]

def inspect_batch(train_loader, num_samples=8):
    """
    检测一个batch的数据
    :param train_loader: 训练数据加载器
    :param num_samples: 要显示的样本数量
    """
    # 获取一个batch的数据
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    print("=== 数据批次信息 ===")
    print(f"Batch大小: {images.shape[0]}")
    print(f"图片形状: {images.shape}")  # [batch_size, channels, height, width]
    print(f"标签形状: {labels.shape}")   # [batch_size]
    print(f"图片数据类型: {images.dtype}")
    print(f"标签数据类型: {labels.dtype}")
    print(f"图片值范围: [{images.min():.4f}, {images.max():.4f}]")
    print(f"标签值范围: [{labels.min()}, {labels.max()}]")
    
    # 统计每个类别的数量
    print("\n=== 当前batch标签分布 ===")
    unique_labels, counts = torch.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label} ({FASHION_MNIST_LABELS[label]}): {count} 个样本")
    
    # 可视化样本
    print(f"\n=== 显示前{num_samples}个样本 ===")
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('FashionMNIST 数据样本检测', fontsize=16)
    
    for i in range(min(num_samples, images.shape[0])):
        row = i // 4
        col = i % 4
        
        # 获取图片和标签
        img = images[i].squeeze()  # 移除channel维度 [1, 224, 224] -> [224, 224]
        label = labels[i].item()
        
        # 显示图片
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'标签: {label}\n({FASHION_MNIST_LABELS[label]})')
        axes[row, col].axis('off')
        
        print(f"样本 {i+1}: 标签={label} ({FASHION_MNIST_LABELS[label]}), 图片形状={img.shape}")
    
    plt.tight_layout()
    plt.show()
    
    return images, labels

def check_data_statistics(train_loader, val_loader):
    """
    检查整个数据集的统计信息
    """
    print("\n=== 数据集统计信息 ===")
    
    # 计算数据集大小
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    print(f"总数据量: {train_size + val_size}")
    print(f"训练/验证比例: {train_size/(train_size+val_size):.2f}/{val_size/(train_size+val_size):.2f}")
    
    # 检查数据加载器设置
    print(f"\n训练DataLoader设置:")
    print(f"  - batch_size: {train_loader.batch_size}")
    print(f"  - shuffle: {train_loader.dataset}")
    print(f"  - 总batch数: {len(train_loader)}")
    
    print(f"\n验证DataLoader设置:")
    print(f"  - batch_size: {val_loader.batch_size}")
    print(f"  - shuffle: False")
    print(f"  - 总batch数: {len(val_loader)}")

def check_data_normalization(images):
    """
    检查数据归一化情况
    """
    print("\n=== 数据归一化检查 ===")
    print(f"图片数据统计:")
    print(f"  - 均值: {images.mean():.4f}")
    print(f"  - 标准差: {images.std():.4f}")
    print(f"  - 最小值: {images.min():.4f}")
    print(f"  - 最大值: {images.max():.4f}")
    
    # 检查是否符合预期的归一化
    expected_mean = (0 - 0.2861) / 0.3530  # (原始值 - mean) / std
    expected_std = 1.0 / 0.3530
    print(f"\n预期归一化结果:")
    print(f"  - 预期均值: {expected_mean:.4f}")
    print(f"  - 预期标准差: {expected_std:.4f}")

if __name__ == "__main__":
    print("🔍 开始检测FashionMNIST数据...")
    
    # 获取数据加载器
    train_loader, val_loader = get_fashion_mnist_dataset()
    
    # 检查数据集统计信息
    check_data_statistics(train_loader, val_loader)
    
    # 检测训练集的一个batch
    print("\n" + "="*50)
    print("🔍 检测训练集batch...")
    train_images, train_labels = inspect_batch(train_loader, num_samples=8)
    
    # 检查数据归一化
    check_data_normalization(train_images)
    
    # 检测验证集的一个batch
    print("\n" + "="*50)
    print("🔍 检测验证集batch...")
    val_images, val_labels = inspect_batch(val_loader, num_samples=8)
    
    print("\n✅ 数据检测完成!")
