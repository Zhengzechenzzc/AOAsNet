import torch
import torch.nn as nn
import torch.distributions as dist


class AoAsm(nn.Module):
    def __init__(self, input_channels, latent_dim=16, batchsize=8):
        """
        捕获注意力层输出与原始特征图因果关系的模块。

        参数:
        - input_channels (int): 输入特征图的通道数。
        - latent_dim (int): VAE的潜在空间维度。
        """
        super(CausalInferenceVAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))  # 将特征降维到 (B, latent_dim, 1, 1)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=3, stride=1, padding=1)
        )

        # StudentT网络
        self.student_t_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 输出 df, loc, scale
        )
        self.df_unconstrained = nn.Parameter(torch.tensor(0.0))

        self.batchsize = batchsize

    def forward(self, original_feature, attention_feature):
        """
        前向传播

        参数:
        - original_feature (torch.Tensor): 原始特征图，形状为 (B, C, H, W)。
        - attention_feature (torch.Tensor): 注意力层的特征图，形状为 (B, C, H, W)。

        返回:
        - torch.Tensor: 融合后的输出特征图。
        """
        # 对原始特征图和注意力特征分别编码
        latent_original = self.encoder(original_feature).squeeze(-1).squeeze(-1)  # (B, latent_dim)
        # if self.enable_intervention and self.intervention_values is not None:
        #     # 如果启用干预，则将潜在特征替换为干预值
        #     latent_attention = self.intervention_values
        latent_attention = self.encoder(attention_feature).squeeze(-1).squeeze(-1)  # (B, latent_dim)

        # 通过Student's T分布计算因果权重
        latent_diff = latent_original - latent_attention  # 差异表示
        loc_scale_df = self.student_t_net(latent_diff)  # (B, 3)
        loc = loc_scale_df[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale_df[..., 1]).clamp(min=1e-3, max=1e6)
        df = nn.functional.softplus(self.df_unconstrained).add(1).expand_as(loc)

        df = df.unsqueeze(-1)  # 将 df 的形状扩展为 [batch_size, 1]
        loc = loc.unsqueeze(-1)  # 将 loc 的形状扩展为 [batch_size, 1]
        scale = scale.unsqueeze(-1)  # 将 scale 的形状扩展为 [batch_size, 1]

        student_t_dist = dist.StudentT(df, loc, scale)

        # print(student_t_dist)
        # 因果权重计算

        causal_weight = student_t_dist.log_prob(latent_diff).exp()  # 权重 (B,)
        causal_weight = causal_weight.view(-1, 1, 1, 1)  # 调整为广播形状

        # 使用VAE解码并融合
        reconstructed_feature = self.decoder(latent_attention.unsqueeze(-1).unsqueeze(-1))

        # 干预
        # if self.enable_intervention:
        #     reconstructed_feature = self.decoder(self.intervention_values.unsqueeze(-1).unsqueeze(-1))
        # else:
        #     reconstructed_feature = self.decoder(latent_attention.unsqueeze(-1).unsqueeze(-1))

        # print(causal_weight.size())
        causal_weight = causal_weight.view(self.batchsize, -1)  # 调整为 [batch_size, latent_dim]
        causal_weight = causal_weight.mean(dim=1, keepdim=True)  # 按 latent_dim 平均，得到 [batch_size, 1]
        causal_weight = causal_weight.view(self.batchsize, 1, 1, 1)  # 调整为 [batch_size, 1, 1, 1]

        # print("causal_weight:", causal_weight.size())
        # print("reconstructed_feature:", reconstructed_feature.size())
        # print("original_feature:", original_feature.size())

        fused_feature = causal_weight * reconstructed_feature + (1 - causal_weight) * original_feature

        return fused_feature

if __name__ == "__main__":

    # 定义输入特征图
    batch_size = 8
    channels = 64
    height, width = 32, 32

    # 原始特征图和注意力特征图
    original_feature = torch.randn(batch_size, channels, height, width)  # 原始特征图
    attention_feature = torch.randn(batch_size, channels, height, width)  # 注意力特征图

    # 初始化 CausalInferenceVAE 模块
    causal_vae = CausalInferenceVAE(input_channels=channels, latent_dim=16)

    # 前向传播
    output_feature = causal_vae(original_feature, attention_feature)

    # 输出结果形状
    print("原始特征图形状:", original_feature.shape)
    print("注意力特征图形状:", attention_feature.shape)
    print("输出特征图形状:", output_feature.shape)

