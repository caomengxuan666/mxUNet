import yaml
from torch.utils.tensorboard import SummaryWriter
from model.Enhanced_Unet import EnhancedUNet
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, config_path="newconfig.yaml"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 加载配置
        self.config = load_config(config_path)
        self.image_ext = self.config['data']['image_ext']
        self.mask_ext = self.config['data']['mask_ext']
        self.channels = self.config['model']['channels']
        self.images = [f for f in os.listdir(image_dir) if f.endswith(self.image_ext)]

        if len(self.images) == 0:
            raise ValueError(f"No valid images found in directory: {self.image_dir}")

        print("当前训练的图像通道数: ", self.channels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(self.image_ext, self.mask_ext)
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # 加载图像和掩膜
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        if self.channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 确保标签是二值化且为 Float 类型
        mask = (mask > 0).float()

        return image, mask


# 自定义损失函数
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(BCEDiceLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, weight=self.weight)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1e-7) / (inputs.sum() + targets.sum() + 1e-7)
        return bce + (1 - dice)


# 数据增强（训练集）
def get_train_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])


# 数据增强（验证集）
def get_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


# 数据加载器准备函数
def prepare_dataloader(config, is_train=True):
    if is_train:
        image_dir = config['data']['image_dir']
        mask_dir = config['data']['mask_dir']
        transform = get_train_transform(config['data']['image_size'])
    else:
        image_dir = config['data']['val_image_dir']
        mask_dir = config['data']['val_mask_dir']
        transform = get_val_transform(config['data']['image_size'])

    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    return DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=is_train)


# 验证阶段
def validate_net(net, val_loader, criterion, device, writer, epoch):
    net.eval()
    val_loss = 0

    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(image)
            label = F.interpolate(label, size=pred.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(pred, label)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    return avg_val_loss


# 训练函数
def train_net(net, device, train_loader, val_loader, args):
    optimizer = optim.Adam(net.parameters(), lr=float(args["learning_rate"]), weight_decay=float(args["weight_decay"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = BCEDiceLoss()
    best_loss = float('inf')

    os.makedirs(args["save_path"], exist_ok=True)
    writer = SummaryWriter(args["log_dir"])

    for epoch in range(args["epochs"]):
        net.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args["epochs"]}', unit='batch') as pbar:
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                pred = net(image)
                label = F.interpolate(label, size=pred.shape[2:], mode="bilinear", align_corners=False)

                loss = criterion(pred, label)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        print(f'Epoch {epoch + 1}/{args["epochs"]} - Loss: {avg_epoch_loss}')

        if (epoch + 1) % args["validate_interval"] == 0:
            avg_val_loss = validate_net(net, val_loader, criterion, device, writer, epoch)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_path = os.path.join(args["save_path"], f'{args["save_name"]}_best.pth')
                torch.save(net.state_dict(), best_model_path)
                print(f"性能最好的模型已保存到: {best_model_path}")

        if (epoch + 1) % args["save_interval"] == 0:
            checkpoint_path = os.path.join(args["save_path"], f'{args["save_name"]}_epoch_{epoch + 1}.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f"模型检查点已保存到: {checkpoint_path}")

    writer.close()


# 加载配置
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# 主函数
def main(config_path='newconfig.yaml'):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = EnhancedUNet(n_channels=config['model']['channels'], n_classes=config['model']['n_classes'])
    net.to(device=device)

    train_loader = prepare_dataloader(config, is_train=True)
    val_loader = prepare_dataloader(config, is_train=False)

    train_net(
        net=net,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        args=config['training']
    )


if __name__ == "__main__":
    main()
