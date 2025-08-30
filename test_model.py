#!/usr/bin/env python3
"""
测试VGG模型的简单脚本
"""
import torch
from model import VGG

def test_model():
    print("=== VGG16-D 模型测试 ===")
    
    # 创建模型
    model = VGG(num_classes=10)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    model.eval()
    test_input = torch.randn(1, 1, 227, 227)  # batch_size=1, channels=1, height=227, width=227
    
    print(f"输入形状: {test_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(test_input)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print("✅ 模型前向传播测试通过!")
        
        # 测试梯度计算
        model.train()
        output = model(test_input)
        loss = torch.sum(output)
        loss.backward()
        print("✅ 模型反向传播测试通过!")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_model()
