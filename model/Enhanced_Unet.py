import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()
        reduced_channels = in_channels // 4
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True)
            )
            for output_size in pool_sizes
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * reduced_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
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
    def __init__(self, in_channels, reduction=16):
        super(MultiScaleAttention, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
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
        self.dropout = nn.Dropout(p=0.1)

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

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
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
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x


class EnhancedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load EfficientNet and modify the first convolutional layer if necessary
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')

        if self.n_channels != 3:
            self.encoder._conv_stem = nn.Conv2d(
                self.n_channels,
                self.encoder._conv_stem.out_channels,
                kernel_size=self.encoder._conv_stem.kernel_size,
                stride=self.encoder._conv_stem.stride,
                padding=self.encoder._conv_stem.padding,
                bias=False
            )

        # Define the initial layers (stem)
        self.inc = nn.Sequential(
            self.encoder._conv_stem,
            self.encoder._bn0,
            self.encoder._swish
        )

        # Define the downsampling blocks by correctly indexing the blocks
        self.down1 = nn.Sequential(*self.encoder._blocks[:2])    # blocks 0-1: output 24 channels
        self.down2 = nn.Sequential(*self.encoder._blocks[2:4])    # blocks 2-3: output 40 channels
        self.down3 = nn.Sequential(*self.encoder._blocks[4:6])    # blocks 4-5: output 80 channels
        self.down4 = nn.Sequential(*self.encoder._blocks[6:16])   # blocks 6-15: output 320 channels

        # Head (after the blocks)
        self.head = nn.Sequential(
            self.encoder._conv_head,  # Conv2d to 1280 channels
            self.encoder._bn1,
            self.encoder._swish
        )

        # Pyramid Pooling and Multi-Scale Attention
        self.ppm = PyramidPooling(1280, pool_sizes=[1, 2, 3, 6])  # 1280 channels
        self.attention = MultiScaleAttention(1280)

        # Decoder
        # Define the decoder 'Up' modules with correct in_channels
        self.up1 = Up(1280 + 80, 512, bilinear)   # x5=1280, x4=80 --> 1280+80=1360
        self.up2 = Up(512 + 40, 256, bilinear)    # x3=40 --> 512+40=552
        self.up3 = Up(256 + 24, 128, bilinear)    # x2=24 --> 256+24=280
        self.up4 = Up(128 + 32, 64, bilinear)     # x1=32 --> 128+32=160

        # Residual Blocks
        self.res_block1 = ResidualBlock(512, 512)
        self.res_block2 = ResidualBlock(256, 256)
        self.res_block3 = ResidualBlock(128, 128)
        self.res_block4 = ResidualBlock(64, 64)

        # Output
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # After inc: 32 channels
        x2 = self.down1(x1)    # After down1: 24 channels
        x3 = self.down2(x2)    # After down2: 40 channels
        x4 = self.down3(x3)    # After down3: 80 channels
        x5 = self.down4(x4)    # After down4: 320 channels
        x5 = self.head(x5)     # After head: 1280 channels

        # Pyramid Pooling and Multi-Scale Attention
        x5 = self.ppm(x5)      # 1280 channels
        x5 = self.attention(x5) # 1280 channels

        # Decoder
        x = self.up1(x5, x4)   # up1: 1280 +80=1360 channels -> 512
        x = self.res_block1(x) # 512 channels
        x = self.up2(x, x3)    # up2: 512 +40=552 channels -> 256
        x = self.res_block2(x) # 256 channels
        x = self.up3(x, x2)    # up3: 256 +24=280 channels -> 128
        x = self.res_block3(x) # 128 channels
        x = self.up4(x, x1)    # up4: 128 +32=160 channels -> 64
        x = self.res_block4(x) # 64 channels

        logits = self.outc(x)  # 1 channel

        return logits
