#!/usr/bin/env python3
"""
简单的数据检测脚本 - 可以在PyCharm中运行
将此代码复制到您的PyCharm中运行
"""

def inspect_data_batch():
    """
    检测一个batch的数据 - 在PyCharm中运行这个函数
    """
    try:
        import torch
        import matplotlib.pyplot as plt
        from dataset import get_fashion_mnist_dataset
        
        # FashionMNIST的类别标签
        LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        print("🔍 开始检测数据...")
        
        # 获取数据加载器
        train_loader, val_loader = get_fashion_mnist_dataset()
        
        # 获取一个batch
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
        
        # 显示前几个样本的信息
        print("\n=== 前8个样本信息 ===")
        for i in range(min(8, len(labels))):
            label = labels[i].item()
            img_shape = images[i].shape
            img_mean = images[i].mean().item()
            print(f"样本 {i+1}: 标签={label} ({LABELS[label]}), 形状={img_shape}, 均值={img_mean:.4f}")
        
        # 统计标签分布
        print("\n=== 当前batch标签分布 ===")
        unique_labels, counts = torch.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"类别 {label} ({LABELS[label]}): {count} 个样本")
        
        # 可视化（如果可能）
        try:
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            fig.suptitle('FashionMNIST 数据样本', fontsize=16)
            
            for i in range(8):
                row = i // 4
                col = i % 4
                
                img = images[i].squeeze()
                label = labels[i].item()
                
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].set_title(f'{label}: {LABELS[label]}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()
            print("✅ 图片显示成功!")
            
        except Exception as e:
            print(f"⚠️ 图片显示失败: {e}")
        
        return images, labels
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保已安装: torch, matplotlib, torchvision")
        return None, None
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return None, None

# 如果直接运行此脚本
if __name__ == "__main__":
    inspect_data_batch()
