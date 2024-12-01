import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, dropout_prob=0.3):
        super(PyramidPooling, self).__init__()
        reduced_channels = in_channels // 4
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            )
            for output_size in pool_sizes
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * reduced_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pooled_features = [
            F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False)
            for stage in self.stages
        ]
        pooled_features.append(x)
        return self.conv(torch.cat(pooled_features, dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(attention))


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, dropout_prob=0.3):
        super(MultiScaleAttention, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.dropout(x)  # Dropout after channel attention
        x = self.spatial_att(x)
        x = self.dropout(x)  # Dropout after spatial attention
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.dropout(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 无需 Dropout
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        self.dropout = nn.Dropout(p=dropout_prob)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout after the first block
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout after adding the shortcut
        return x


class EnhancedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 硬编码使用 'efficientnet-b4'
        efficientnet_version = 'efficientnet-b4'

        # 加载 EfficientNet-b4 并修改第一层卷积（如果输入通道数 != 3）
        self.encoder = EfficientNet.from_pretrained(efficientnet_version)

        if self.n_channels != 3:
            self.encoder._conv_stem = nn.Conv2d(
                self.n_channels,
                self.encoder._conv_stem.out_channels,
                kernel_size=self.encoder._conv_stem.kernel_size,
                stride=self.encoder._conv_stem.stride,
                padding=self.encoder._conv_stem.padding,
                bias=False
            )

        # 定义初始层（stem）
        self.inc = nn.Sequential(
            self.encoder._conv_stem,
            self.encoder._bn0,
            self.encoder._swish
        )

        # 定义下采样块，根据 EfficientNet-b4 的实际架构调整块的范围
        # EfficientNet-b4 的阶段：
        # 阶段1: Block 0-4 → output_filters=32
        # 阶段2: Block 5-9 → output_filters=56
        # 阶段3: Block 10-16 → output_filters=160
        # 阶段4: Block 17-21 → output_filters=160

        # 设置下采样块
        self.down1 = nn.Sequential(*self.encoder._blocks[:5])    # Blocks 0-4: output_filters=32
        self.down2 = nn.Sequential(*self.encoder._blocks[5:10])  # Blocks 5-9: output_filters=56
        self.down3 = nn.Sequential(*self.encoder._blocks[10:17]) # Blocks 10-16: output_filters=160
        self.down4 = nn.Sequential(*self.encoder._blocks[17:22])  # Blocks 17-21: output_filters=160

        # 获取下采样块的输出通道数
        down1_out_channels = self.down1[-1]._block_args.output_filters  # 32
        down2_out_channels = self.down2[-1]._block_args.output_filters  # 56
        down3_out_channels = self.down3[-1]._block_args.output_filters  # 160
        down4_out_channels = self.down4[-1]._block_args.output_filters  # 160

        # 打印以确认输出通道数
        print(f"down1_out_channels: {down1_out_channels}")  # 32
        print(f"down2_out_channels: {down2_out_channels}")  # 56
        print(f"down3_out_channels: {down3_out_channels}")  # 160
        print(f"down4_out_channels: {down4_out_channels}")  # 160

        # Head（在块之后）
        self.head = nn.Sequential(
            nn.Conv2d(down4_out_channels, 1792, kernel_size=1, stride=1, bias=False),  # 160 → 1792
            nn.BatchNorm2d(1792),
            nn.SiLU()
        )

        # Pyramid Pooling 和 Multi-Scale Attention
        self.ppm_in_channels = 1792
        self.ppm = PyramidPooling(self.ppm_in_channels, pool_sizes=[1, 2, 3, 6])
        self.attention = MultiScaleAttention(self.ppm_in_channels)

        # 解码器
        self.up1 = Up(self.ppm_in_channels + down4_out_channels, 512, bilinear)   # 1792 +160=1952 →512
        self.up2 = Up(512 + down2_out_channels, 256, bilinear)                    # 512 +56=568 →256
        self.up3 = Up(256 + down1_out_channels, 128, bilinear)                    # 256 +32=288 →128
        self.up4 = Up(128 + self.inc[0].out_channels, 64, bilinear)               # 128 +32=160 →64

        # 残差块
        self.res_block1 = ResidualBlock(512, 512)
        self.res_block2 = ResidualBlock(256, 256)
        self.res_block3 = ResidualBlock(128, 128)
        self.res_block4 = ResidualBlock(64, 64)

        # 输出层
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.head(x5)

        # Pyramid Pooling 和 Multi-Scale Attention
        x5 = self.ppm(x5)
        x5 = self.attention(x5)

        # 解码器
        x = self.up1(x5, x4)
        x = self.res_block1(x)
        x = self.up2(x, x3)
        x = self.res_block2(x)
        x = self.up3(x, x2)
        x = self.res_block3(x)
        x = self.up4(x, x1)
        x = self.res_block4(x)

        logits = self.outc(x)
        return logits

