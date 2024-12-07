from .Enhanced_Unet import *

class EnhancedUNetOptimized(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EnhancedUNetOptimized, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器部分
        efficientnet_version = 'efficientnet-b4'
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

        # 编码器各阶段
        self.inc = nn.Sequential(
            self.encoder._conv_stem,
            self.encoder._bn0,
            self.encoder._swish
        )
        self.down1 = nn.Sequential(*self.encoder._blocks[:5])    # Blocks 0-4
        self.down2 = nn.Sequential(*self.encoder._blocks[5:10])  # Blocks 5-9
        self.down3 = nn.Sequential(*self.encoder._blocks[10:17]) # Blocks 10-16
        self.down4 = nn.Sequential(*self.encoder._blocks[17:22]) # Blocks 17-21

        # 通道数
        down1_out_channels = 48   # 第1阶段
        down2_out_channels = 32   # 第2阶段
        down3_out_channels = 56   # 第3阶段
        down4_out_channels = 160  # 第4阶段

        # 修正解码器输入通道数
        self.up1 = Up(down4_out_channels + down4_out_channels, 160, bilinear)  # 拼接 160+160
        self.up2 = Up(160 + down3_out_channels, 160, bilinear)                 # 拼接 160+56
        self.up3 = Up(160 + down2_out_channels, 80, bilinear)                  # 拼接 160+32
        self.up4 = Up(80 + down1_out_channels, 40, bilinear)                   # 拼接 80+48

        # 输出层
        self.outc = OutConv(40, n_classes)

    def forward(self, x):
        # 编码器部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 打印形状
        #print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}, x3 shape: {x3.shape}, x4 shape: {x4.shape}, x5 shape: {x5.shape}")

        # 解码器部分
        x = self.up1(x5, x4)
        #print(f"up1 output shape: {x.shape}")
        x = self.up2(x, x3)
        #print(f"up2 output shape: {x.shape}")
        x = self.up3(x, x2)
        #print(f"up3 output shape: {x.shape}")
        x = self.up4(x, x1)
        #print(f"up4 output shape: {x.shape}")

        logits = self.outc(x)
        return logits




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例并将其移动到设备
    model = EnhancedUNetOptimized(n_channels=3, n_classes=1).to(device)

    # 创建输入数据并将其移动到设备
    x = torch.randn(1, 3, 256, 256).to(device)

    # 打印模型摘要
    summary(model, (3, 256, 256))