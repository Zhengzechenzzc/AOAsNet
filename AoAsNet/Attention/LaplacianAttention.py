import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianAttention(nn.Module):
    def __init__(self, input_channels):
        """
        拉普拉斯域注意力机制

        参数:
        - input_channels (int): 输入特征图的通道数
        """
        super(LaplacianAttention, self).__init__()
        # 定义拉普拉斯卷积核
        kernel = torch.tensor([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.laplacian_kernel = nn.Parameter(kernel, requires_grad=False)
        self.input_channels = input_channels

        # 通道权重生成器
        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // 4, input_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        参数:
        - x (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)

        返回:
        - torch.Tensor: 加权后的特征图，形状与输入一致
        """
        # 拉普拉斯特征提取
        batch_size, channels, height, width = x.size()
        laplacian = F.conv2d(x, self.laplacian_kernel.repeat(channels, 1, 1, 1),
                             padding=1, groups=channels)

        # 全局平均池化计算通道权重
        pooled = F.adaptive_avg_pool2d(laplacian, 1).view(batch_size, channels)
        weights = self.fc(pooled).view(batch_size, channels, 1, 1)

        # 加权原始特征图
        output = x * (weights+1)
        return output


# 示例用法
if __name__ == "__main__":
    # 模拟输入特征图 (batch_size, channels, height, width)
    input_tensor = torch.randn(8, 64, 32, 32)  # (batch_size=8, channels=64, height=32, width=32)

    # 创建拉普拉斯注意力模块
    laplacian_attention = LaplacianAttention(input_channels=64)

    # 前向传播
    output_tensor = laplacian_attention(input_tensor)

    print("输入特征图形状:", input_tensor.shape)
    print("输出特征图形状:", output_tensor.shape)
