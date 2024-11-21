from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        pooled_features = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
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
        self.dropout = nn.Dropout(p=0.3)  # 添加 Dropout 防止过拟合

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.dropout(x)  # Dropout 添加到 Up 模块


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EnhancedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ResNet50 as Encoder
        self.encoder = resnet50(pretrained=True)
        self.encoder_layers = list(self.encoder.children())

        self.inc = nn.Sequential(*self.encoder_layers[:4])  # Conv1
        self.down1 = nn.Sequential(*self.encoder_layers[4])  # Layer1
        self.down2 = self.encoder_layers[5]  # Layer2
        self.down3 = self.encoder_layers[6]  # Layer3
        self.down4 = self.encoder_layers[7]  # Layer4

        # Pyramid Pooling Module
        self.ppm = PyramidPooling(2048, pool_sizes=[1, 2, 3, 6])

        # Decoder
        self.up1 = Up(2048 + 1024, 512, bilinear)
        self.up2 = Up(512 + 512, 256, bilinear)
        self.up3 = Up(256 + 256, 128, bilinear)
        self.up4 = Up(128 + 64, 64, bilinear)

        # Attention Modules
        self.channel_att = ChannelAttention(2048)
        self.spatial_att = SpatialAttention()

        # Output
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # PPM + Attention
        x5 = self.ppm(x5)
        x5 = self.channel_att(x5)
        x5 = self.spatial_att(x5)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits
