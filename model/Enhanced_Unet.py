from .unet_parts import *
from torchvision.models import resnet34


class PyramidPooling(nn.Module):
    """金字塔池化模块 (Pyramid Pooling Module)"""

    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for output_size in pool_sizes
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // len(pool_sizes) * len(pool_sizes), in_channels, kernel_size=1),
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


class EnhancedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = resnet34(pretrained=True)
        self.encoder_layers = list(self.encoder.children())
        self.inc = nn.Sequential(*self.encoder_layers[:3])  # Conv1
        self.down1 = nn.Sequential(*self.encoder_layers[3:5])  # Layer1
        self.down2 = self.encoder_layers[5]  # Layer2
        self.down3 = self.encoder_layers[6]  # Layer3
        self.down4 = self.encoder_layers[7]  # Layer4

        # 调整金字塔池化的通道数
        self.ppm = PyramidPooling(512, pool_sizes=[1, 2, 3, 6])  # ResNet 最后一层输出 512 通道

        # Decoder
        self.up1 = Up(512 + 256, 256, bilinear)  # ResNet block 输出通道调整为 512
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.up4 = Up(64 + 64, 64, bilinear)

        # Attention Modules
        self.channel_att = ChannelAttention(512)
        self.spatial_att = SpatialAttention()

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

        # 调整输出尺寸与输入一致
        logits = self.outc(x)
        logits = F.interpolate(logits, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        return logits