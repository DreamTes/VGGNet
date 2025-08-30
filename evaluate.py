import torch

from model import VGG
from dataset import get_test_fashion_mnist_dataset

def evaluate_model(model, test_loader, device):

    model = model.to(device)

    test_current = 0
    test_total = 0

    with torch.inference_mode():
        for test_data, test_target in test_loader:
            test_data, test_target = test_data.to(device), test_target.to(device)
            model.eval()  # 设置模型为评估模式
            outputs = model(test_data)
            pre_lab = torch.argmax(outputs, dim=1)  # 获取预测标签
            test_current += (pre_lab == test_target).sum().item()
            test_total += test_target.size(0)

    test_accuracy = test_current / test_total
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG()
    model_path = './checkpoints/best_model_20250826_184552.pth'  # 模型文件路径
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)  # 加载模型参数

    test_loader = get_test_fashion_mnist_dataset()
    evaluate_model(model, test_loader, device)  # 评估模型性能

    # # 可选：逐样本打印预测与标签
    # with torch.inference_mode():
    #     model.eval()
    #     for test_data, test_target in test_loader:
    #         test_data, test_target = test_data.to(device), test_target.to(device)
    #         outputs = model(test_data)
    #         pred_labels = torch.argmax(outputs, dim=1)
    #         for pred, true in zip(pred_labels.view(-1).cpu().tolist(), test_target.view(-1).cpu().tolist()):
    #             print(f"Predicted: {pred}, Actual: {true}")


