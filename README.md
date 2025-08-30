## 项目简介

基于 PyTorch 的 VGG16-D 模型，在 FashionMNIST（10 类灰度图）上进行训练、验证与评估。项目包含数据可视化、训练曲线保存、单 batch 过拟合自检、模型权重初始化等实用功能。

- **数据集**: FashionMNIST（自动下载）
- **输入尺寸**: 1×224×224（灰度单通道，已在 `dataset.py` 中 resize）
- **模型**: VGG16-D（5 个卷积块 + 3 个全连接层）
- **任务**: 10 分类

---

## 环境准备

- Python 3.8+
- 建议使用虚拟环境（venv/conda）

```bash
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib pandas torchsummary
```

> 如果你使用 GPU，请根据 PyTorch 官网选择与你的 CUDA 版本匹配的安装命令。

---

## 代码结构

- `model.py`：VGG16-D 模型与权重初始化
- `dataset.py`：数据集加载与预处理（训练/验证/测试）
- `train.py`：训练主流程、训练曲线绘制、单 batch 过拟合自检
- `evaluate.py`：加载最佳权重进行测试集评估
- `inspect_data.py`：取出一个 batch 的图像与标签进行可视化与统计
- `checkpoints/`：训练时按时间戳保存的最佳权重
- `results/`：训练曲线图（Loss/Accuracy）

---

## 数据与预处理

`dataset.py`：
- 自动下载 FashionMNIST 到 `./data`
- 统一 resize 到 224×224
- 标准化到 mean=0.2861、std=0.3530（常用配置）
- 训练/验证默认划分比例为 0.8/0.2
- 默认 `batch_size=64`

---

## 模型说明（VGG16-D）

- 输入：`[N, 1, 224, 224]`
- 卷积块（每个块后 MaxPool2d(kernel=2, stride=2)）：
  - Block1: 64×3×3 ×2
  - Block2: 128×3×3 ×2
  - Block3: 256×3×3 ×3
  - Block4: 512×3×3 ×3
  - Block5: 512×3×3 ×3
- 池化 5 次后，特征图尺寸 7×7，通道 512 → Flatten 后全连接：
  - FC: 512×7×7 → 4096 → 4096 → num_classes(10)
- 激活：ReLU
- Dropout：0.5（在全连接层部分）

### 权重初始化
在 `model.py` 中：
- Conv2d / Linear：Kaiming Normal，`mode='fan_out'`，`nonlinearity='relu'`
- Bias：常数 0

作用：匹配 ReLU，保持前向/反向的方差更稳定，减小梯度消失/爆炸风险，加快收敛。

---

## 训练

直接运行：
```bash
python train.py
```
功能：
- 使用 CrossEntropyLoss
- 优化器/调度器（可在 `train.py` 中调节）：
  - 示例：SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)，StepLR(step_size=10, gamma=0.1)
- 训练日志打印每个 epoch 的 Train/Val Loss 与 Accuracy
- 在 `results/` 下保存训练曲线图（带时间戳）
- 在 `checkpoints/` 下保存最佳模型权重（验证准确率最高）

> 如果显存/收敛不稳定：尝试减小 `batch_size`，或降低初始 LR（如 0.005），或临时关闭/减小 Dropout。

---

## 单 Batch 过拟合自检（强烈推荐）

用于快速验证“模型和训练流程是否正常工作”，目标是把同一 batch 的 loss 压到非常低、acc 接近 100%。

在 `train.py` 中提供：
```python
from dataset import get_fashion_mnist_dataset
from model import VGG
from train import sanity_overfit_one_batch
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG().to(device)
train_loader, _ = get_fashion_mnist_dataset()
sanity_overfit_one_batch(model, train_loader, device)
```

我们在实现中使用了：
- 较小 LR（如 0.02 的 SGD）
- 可选的梯度裁剪（clip_grad_norm_）防止局部数值爆炸

如果单 batch 也无法收敛，请重点检查：
- 标签是否为 LongTensor 且范围 0~9
- 输入是否为 `[N, 1, 224, 224]`
- 学习率是否过高导致发散
- 权重初始化是否合理（当前已为 Kaiming + ReLU）

---

## 可视化与数据检查

- 训练曲线：训练结束后自动保存到 `results/train_curves_*.png`
- 取 batch 看图与标签：
```bash
python inspect_data.py
```
将输出 batch 的尺寸、数值范围、标签分布，并展示前若干张样本图。

---

## 评估

`evaluate.py` 会加载指定的权重并在测试集上评估：
```bash
python evaluate.py
```
如需自定义权重路径，修改 `evaluate.py` 中的 `model_path`。

---

## 常见问题（FAQ）

- Q: 训练时 loss 一直 ~2.3026，acc ~0.1？
  - A: 这是 10 类随机猜测的典型表现。请做单 batch 自检；常见原因是学习率过大/优化器不当/初始化不匹配/Dropout 干扰等。

- Q: CUDA 报错或显存不足？
  - A: 降低 batch_size；或在 CPU 上先验证流程；或减少模型规模（如先用更浅的 CNN 测试）。

- Q: 训练很慢？
  - A: 开启 cudnn benchmark（可在主入口加上 `torch.backends.cudnn.benchmark = True`），并确保使用 GPU。

---

## 致谢

- VGG: Very Deep Convolutional Networks for Large-Scale Image Recognition (Simonyan & Zisserman)
- PyTorch 与 TorchVision 社区贡献的开源实现与数据集支持
