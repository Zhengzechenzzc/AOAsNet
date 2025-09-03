import torch
import torch.nn as nn



class AttentionFramework(nn.Module):
    def __init__(self, attention_modules, input_channels=64, weights=None):
        """
        注意力机制框架

        参数:
        - attention_modules (list of nn.Module): 注意力机制模块列表，每个模块输入和输出形状一致。
        - input_channels (int): 输入通道数。
        - weights (list of float): 每个注意力模块的权重列表，默认为均等权重。
        """
        super(AttentionFramework, self).__init__()
        self.attention_modules = nn.ModuleList(attention_modules)
        self.num_modules = len(attention_modules)
        self.weights = nn.Parameter(torch.ones(self.num_modules) if weights is None else torch.tensor(weights, dtype=torch.float32), requires_grad=True)
        self.input_channels = input_channels

    def forward(self, x, selected_modules=None):
        """
        前向传播

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, input_channels, height, width)。
        - selected_modules (list of int): 选择应用的注意力模块索引列表，如果为None，则使用所有模块。

        返回:
        - torch.Tensor: 形状与输入一致的输出张量。
        """
        if selected_modules is None:
            selected_modules = list(range(self.num_modules))

        outputs = []
        for idx in selected_modules:
            attention_output = self.attention_modules[idx](x)
            outputs.append(self.weights[idx] * attention_output)

        # 汇总所有选定注意力机制的输出
        output = torch.stack(outputs, dim=0).sum(dim=0)
        return output


# 示例注意力模块
class SampleAttentionModule(nn.Module):
    def __init__(self, input_channels):
        super(SampleAttentionModule, self).__init__()
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(input_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.bn(attention)
        attention = self.act(attention)
        return attention * x


# 测试代码
if __name__ == "__main__":
    # 定义5个不同的注意力模块
    input_channels = 64
    attention_modules = [
        SampleAttentionModule(input_channels),
        SampleAttentionModule(input_channels),
        SampleAttentionModule(input_channels),
        SampleAttentionModule(input_channels),
        SampleAttentionModule(input_channels)
    ]

    # 初始化框架
    framework = AttentionFramework(attention_modules, input_channels)

    # 测试输入
    x = torch.randn(8, input_channels, 32, 32)  # (batch_size, channels, height, width)

    # 选择模块 [0, 2, 4] 并计算结果
    selected_modules = [0, 2, 4]
    output = framework(x, selected_modules)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
