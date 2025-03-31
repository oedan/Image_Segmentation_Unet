""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from .unet_parts import * 


class InceptionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        初始化 InceptionUNet 模型，一个结合了 Inception 模块的 U-Net 变体。

        参数:
            n_channels (int): 输入图像的通道数
            n_classes (int): 输出类别的数量
            bilinear (bool): 是否使用双线性插值进行上采样（若为 False，则使用转置卷积）
        """
        super(InceptionUNet, self).__init__()
        self.n_channels = n_channels  # 保存输入通道数
        self.n_classes = n_classes    # 保存输出类别数
        self.bilinear = bilinear      # 保存上采样方式的选择

        # 定义 Inception 模块，用于增强特征提取能力
        self.block1 = InceptionConv(64, 32)    # 输入 64 通道，输出 32 通道
        self.block2 = InceptionConv(128, 64)   # 输入 128 通道，输出 64 通道
        self.block3 = InceptionConv(256, 128)  # 输入 256 通道，输出 128 通道
        self.block4 = InceptionConv(512, 128)  # 输入 512 通道，输出 128 通道

        # U-Net 的编码器部分（下采样路径）
        self.inc = DoubleConv(n_channels, 64)  # 输入层：初始双卷积块
        self.down1 = Down(64, 128)             # 下采样：64 -> 128 通道
        self.down2 = Down(128, 256)            # 下采样：128 -> 256 通道
        self.down3 = Down(256, 512)            # 下采样：256 -> 512 通道
        factor = 2 if bilinear else 1          # 根据上采样方式调整通道数缩放因子
        self.down4 = Down(512, 1024 // factor) # 最深层下采样：512 -> 1024/factor 通道

        # U-Net 的解码器部分（上采样路径），结合 Inception 输出
        self.up1 = UpInception(1024 + 512, 256 // factor, bilinear)  # 上采样：连接 x5 和 x4
        self.up2 = UpInception(896, 128 // factor, bilinear)         # 上采样：连接上一层和 x3
        self.up3 = UpInception(448, 32 // factor, bilinear)          # 上采样：连接上一层和 x2
        self.up4 = UpInception(208, 16, bilinear)                    # 上采样：连接上一层和 x1
        self.outc = OutConv(16, n_classes)                           # 输出层：生成最终分割图

    def forward(self, x):
        """
        定义模型的前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, n_channels, height, width)

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, n_classes, height, width)
        """
        # 编码器路径：逐步下采样提取特征
        x1 = self.inc(x)       # 输入层处理，生成初始特征图
        x2 = self.down1(x1)    # 第一次下采样
        x3 = self.down2(x2)    # 第二次下采样
        x4 = self.down3(x3)    # 第三次下采样
        x5 = self.down4(x4)    # 第四次下采样，到达网络底部

        # Inception 模块路径：从每一层提取多尺度特征
        block1 = self.block1(x1)  # 对 x1 应用 Inception 模块
        block2 = self.block2(block1)  # 对 block1 应用 Inception 模块
        block3 = self.block3(block2)  # 对 block2 应用 Inception 模块
        block4 = self.block4(block3)  # 对 block3 应用 Inception 模块

        # 解码器路径：逐步上采样并融合特征
        x = self.up1(x5, x4, block4)  # 上采样并融合 x5、x4 和 block4
        # x = torch.cat(x, block4)    # （注释掉的代码）直接拼接，未使用
        x = self.up2(x, x3, block3)   # 上采样并融合上一层、x3 和 block3
        # x = torch.cat(x, block3)    # （注释掉的代码）直接拼接，未使用
        x = self.up3(x, x2, block2)   # 上采样并融合上一层、x2 和 block2
        # x = torch.cat(x, block2)    # （注释掉的代码）直接拼接，未使用
        x = self.up4(x, x1, block1)   # 上采样并融合上一层、x1 和 block1
        # x = torch.cat(x, block1)    # （注释掉的代码）直接拼接，未使用

        # 输出层处理
        x = self.outc(x)              # 生成最终分割图
        x = torch.sigmoid(x)          # 应用 sigmoid 激活，输出概率值（适用于二分类或多标签任务）
        return x
