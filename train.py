import copy
import time

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os

from model import VGG
from dataset import get_fashion_mnist_dataset

EPOCHS = 20
LEARNING_RATE = 0.001  # 提高学习率，VGG网络需要相对较高的学习率

def train_model_process(model, train_loader, val_loader, epochs, learning_rate):
    """
    训练模型的主流程
    :param model: PyTorch模型
    :param train_loader: 训练集DataLoader
    :param epochs: 训练轮数
    :param learning_rate: 学习率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"当前使用的设备是: {device}")  # 打印当前使用的设备

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
   #  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=0)
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #复制当前参数
    best_model_state=copy.deepcopy(model.state_dict())

    #初始化参数
    best_accuracy = 0.0 # 最佳验证集准确率
    train_loss_all = [] # 训练集损失列表
    val_loss_all = [] # 验证集损失列表
    train_accuracy_all = [] # 训练集准确率列表
    val_accuracy_all = [] # 验证集准确率列表
    since = time.time() # 记录训练开始时间
    # 要做单batch过拟合检查，请调用 sanity_overfit_one_batch(model, train_loader, device) 再返回
    
    # 训练过程
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")# 打印当前轮数
        print('-' * 10) # 分隔线
        #初始化参数
        train_loss = 0.0 # 训练集损失
        train_correct = 0 # 训练集正确预测数量（累加为 Python 标量）
        val_loss = 0.0 # 验证集损失
        val_correct = 0 # 验证集正确预测数量（累加为 Python 标量）
        train_num = 0 # 训练集样本数量
        val_num = 0 # 验证集样本数量
        model.train()  # 设置模型为训练模式

        # 对每一个mini-batch进行训练和计算
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 清零梯度
            outputs = model(data)  # 前向传播
            pre_lab = torch.argmax(outputs, dim=1)  # 获取预测标签
            loss = criterion(outputs, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item()* data.size(0)  # 累加损失
            train_correct += (pre_lab == target).sum().item()  # 累加正确预测数量
            train_num += data.size(0)


        # 在验证集上评估模型性能
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                outputs = model(data)  # 前向传播
                pre_lab = torch.argmax(outputs, dim=1)  # 获取预测标签
                loss = criterion(outputs, target)  # 计算损失
                val_loss += loss.item() * data.size(0)  # 累加损失
                val_correct += (pre_lab == target).sum().item()  # 累加正确预测数量
                val_num += data.size(0) # 累加样本数量

        train_loss_all.append(float(train_loss / train_num)) # 计算并保存训练集损失
        train_accuracy_all.append(train_correct / train_num) # 计算并保存训练集准确率
        val_loss_all.append(float(val_loss / val_num)) # 计算并保存验证集损失
        val_accuracy_all.append(val_correct / val_num) # 计算并保存验证集准确率

        print(f"Train Loss: {train_loss_all[-1]:.4f}, Train Accuracy: {train_accuracy_all[-1]:.4f}")
        print(f"Val Loss: {val_loss_all[-1]:.4f}, Val Accuracy: {val_accuracy_all[-1]:.4f}")

        # 每个 epoch 结束后，保存最优模型参数
        if val_accuracy_all[-1] > best_accuracy:
            best_accuracy = val_accuracy_all[-1]
            best_model_state = copy.deepcopy(model.state_dict())
        
        # 更新学习率
        scheduler.step()

    time_used = time.time() - since  # 计算训练时间
    print(f"训练和验证耗费的时间： {time_used // 60:.0f}m {time_used % 60:.0f}s")

    os.makedirs('checkpoints', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join('checkpoints', f'best_model_{timestamp}.pth')
    torch.save(best_model_state, save_path)  # 保存最佳模型参数

    train_process = pd.DataFrame(data={
        'epoch': range(epochs),
        'train_loss': train_loss_all,
        'val_loss': val_loss_all,
        'train_accuracy': train_accuracy_all,
        'val_accuracy': val_accuracy_all
    })

    return train_process


def matplot_acc_loss(train_process):
    """
    绘制训练和验证的损失和准确率曲线
    :param train_process: 训练过程DataFrame
    """
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process['train_loss'], label='Train Loss')
    plt.plot(train_process['epoch'], train_process['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_accuracy'], label='Train Accuracy')
    plt.plot(train_process['epoch'], train_process['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # 保存到 results 目录，文件名带时间戳
    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join('results', f'train_curves_{timestamp}.png')
    plt.savefig(save_path, dpi=150)

    # 同时仍然显示
    plt.show()

def sanity_overfit_one_batch(model, train_loader, device):
    """单batch过拟合检查：应当能迅速把loss压到接近0、acc接近1.0"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    # 使用更小的学习率，并启用梯度裁剪，避免数值爆炸
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=0)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # 关闭dropout影响：eval只对Dropout有效，BN若有需保留train；本模型无BN
    model.train()
    for step in range(200):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            print(f"loss 非有限值: {loss.item()}, 终止")
            break
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if (step + 1) % 50 == 0:
            acc = (logits.argmax(1) == labels).float().mean().item()
            print(f"sanity step {step+1}: loss={loss.item():.4f}, acc={acc:.3f}")

if __name__ == "__main__":
    model = VGG()
    train_loader, val_loader = get_fashion_mnist_dataset()
    training_history = train_model_process(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
    matplot_acc_loss(training_history)  # 绘制损失和准确率曲线
    
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = VGG().to(device)
    # train_loader, _ = get_fashion_mnist_dataset()
    # sanity_overfit_one_batch(model, train_loader, device)
