import torch
import torch.nn as nn
import torch.nn.functional as F

class GLCMAttention(nn.Module):
    def __init__(self, input_channels, gray_levels=8, distances=[1], angles=[0, torch.pi/4, torch.pi/2, 3*torch.pi/4]):
        """
        基于 GLCM 的注意力机制 (PyTorch 实现)
        参数:
        - input_channels (int): 输入特征图的通道数
        - gray_levels (int): 灰度级别，用于量化输入特征图
        - distances (list): 距离列表，用于计算 GLCM
        - angles (list): 角度列表，用于计算 GLCM
        """
        super(GLCMAttention, self).__init__()
        self.input_channels = input_channels
        self.gray_levels = gray_levels
        self.distances = distances
        self.angles = angles

        # 通道权重生成器
        self.fc = nn.Sequential(
            nn.Linear(len(distances) * len(angles) * 4, input_channels // 4, bias=False),  # 使用 4 个特征
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // 4, input_channels, bias=False),
            nn.Sigmoid()
        )

    def compute_glcm(self, x, distance, angle):
        """
        使用 PyTorch 计算 GLCM
        参数:
        - x (torch.Tensor): 单通道特征图，形状为 (batch_size, 1, height, width)
        - distance (int): 像素之间的距离
        - angle (float): 像素之间的方向角度（弧度制）
        返回:
        - torch.Tensor: GLCM 矩阵，形状为 (batch_size, gray_levels, gray_levels)
        """
        batch_size, _, height, width = x.size()
        dx = int(round(distance * torch.cos(angle)))
        dy = int(round(-distance * torch.sin(angle)))

        # 偏移计算，处理边界条件
        padded = F.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)), mode='constant', value=0)
        shifted = padded[:, :, max(-dy, 0):height + max(-dy, 0), max(-dx, 0):width + max(-dx, 0)]

        # 量化到灰度级别
        x_quantized = (x * (self.gray_levels - 1)).long()
        shifted_quantized = (shifted * (self.gray_levels - 1)).long()

        # 构造 GLCM
        glcm = torch.zeros((batch_size, self.gray_levels, self.gray_levels), device=x.device)
        for i in range(self.gray_levels):
            for j in range(self.gray_levels):
                mask = (x_quantized == i) & (shifted_quantized == j)
                glcm[:, i, j] = mask.view(batch_size, -1).sum(dim=1)

        return glcm

    def extract_features(self, glcm):
        """
        从 GLCM 中提取纹理特征
        参数:
        - glcm (torch.Tensor): GLCM 矩阵，形状为 (batch_size, gray_levels, gray_levels)
        返回:
        - torch.Tensor: 提取的特征，形状为 (batch_size, 4)
        """
        # 归一化
        glcm = glcm / (glcm.sum(dim=(1, 2), keepdim=True) + 1e-6)

        # 计算特征
        i = torch.arange(self.gray_levels, device=glcm.device).view(1, -1, 1)
        j = torch.arange(self.gray_levels, device=glcm.device).view(1, 1, -1)
        contrast = ((i - j) ** 2 * glcm).sum(dim=(1, 2))
        homogeneity = (glcm / (1.0 + (i - j).abs())).sum(dim=(1, 2))
        energy = (glcm ** 2).sum(dim=(1, 2))
        correlation = (((i - i.mean()) * (j - j.mean()) * glcm).sum(dim=(1, 2)) /
                       (i.std() * j.std() + 1e-6))

        return torch.stack([contrast, homogeneity, energy, correlation], dim=1)

    def forward(self, x):
        """
        前向传播
        参数:
        - x (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
        返回:
        - torch.Tensor: 加权后的特征图
        """
        batch_size, channels, height, width = x.size()
        weights = torch.zeros(batch_size, channels, device=x.device)

        for c in range(channels):
            channel_feature = x[:, c:c + 1, :, :]  # 单通道特征
            all_features = []

            for distance in self.distances:
                for angle in self.angles:
                    glcm = self.compute_glcm(channel_feature, distance, angle)
                    features = self.extract_features(glcm)
                    all_features.append(features)

            all_features = torch.cat(all_features, dim=1)
            weights[:, c] = self.fc(all_features).squeeze()

        weights = weights.view(batch_size, channels, 1, 1)
        output = x * (weights+1)
        return output


# 示例用法
if __name__ == "__main__":
    # 模拟输入特征图 (batch_size, channels, height, width)
    input_tensor = torch.randn(8, 64, 32, 32).cuda()  # (batch_size=8, channels=64, height=32, width=32)

    # 创建 GLCM 注意力模块
    glcm_attention = GLCMAttention(input_channels=64).cuda()

    # 前向传播
    output_tensor = glcm_attention(input_tensor)

    print("输入特征图形状:", input_tensor.shape)
    print("输出特征图形状:", output_tensor.shape)
