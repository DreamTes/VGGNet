from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as data

def get_fashion_mnist_dataset(root_dir='./data',train_split_ratio=0.8):
    """
    获取FashionMNIST数据集，并划分为训练集和验证集。

    :param root_dir: 数据集存储路径
    :param train_split_ratio: 训练集占总数据集的比例
    :return: 训练集和验证集的DataLoader
    """
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize(224),  # 确保图片尺寸是224x224
        transforms.ToTensor(),  # 将图片转换为PyTorch张量
        transforms.Normalize((0.2861,), (0.3530,))  # 标准化（FashionMNIST 常用均值/方差）
    ])

    # 下载并加载FashionMNIST数据集
    full_train_dataset = FashionMNIST(root=root_dir, train=True, download=True, transform=transform)

    # 划分训练集和验证集
    train_size = int(train_split_ratio * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_data, val_data = data.random_split(full_train_dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(dataset=val_data, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, val_loader

# 获取测试集
def get_test_fashion_mnist_dataset(root_dir='./data'):
    
    test_dataset = FashionMNIST(
        root=root_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.2861,), (0.3530,))
        ])
    )

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    return test_loader
