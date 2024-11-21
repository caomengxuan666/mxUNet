import yaml
from torch.utils.tensorboard import SummaryWriter
from model.unet_model import UNet
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, config_path="config.yaml"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        #读取配置
        self.config = load_config(config_path)
        self.image_ext = self.config['data']['image_ext']
        self.mask_ext = self.config['data']['mask_ext']
        self.channels = self.config['model']['channels']
        self.images = [f for f in os.listdir(image_dir) if f.endswith(self.image_ext)]

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

        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        # 根据配置文件中的 channels 参数决定是否将图像转换为灰度图
        if self.channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def compute_metrics(pred, label):
    """
    计算多个指标: Dice, IoU, Precision, Recall, F1 Score, Specificity, Pixel Accuracy, MAE
    """
    pred = torch.sigmoid(pred) > 0.5  # 二值化
    pred = pred.view(-1).float()
    label = label.view(-1).float()

    intersection = (pred * label).sum()
    union = pred.sum() + label.sum() - intersection
    TP = intersection  # True Positives
    FP = pred.sum() - intersection  # False Positives
    FN = label.sum() - intersection  # False Negatives
    TN = (1 - pred).sum() - FN  # True Negatives

    # 计算指标
    dice = (2. * TP) / (2. * TP + FP + FN + 1e-7)  # Dice Coefficient
    iou = TP / (union + 1e-7)  # Intersection over Union
    mae = torch.abs(pred - label).mean()  # Mean Absolute Error
    precision = TP / (TP + FP + 1e-7)  # Precision
    recall = TP / (TP + FN + 1e-7)  # Recall
    f1 = 2 * precision * recall / (precision + recall + 1e-7)  # F1 Score
    specificity = TN / (TN + FP + 1e-7)  # Specificity
    pixel_accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-7)  # Pixel Accuracy

    return {
        "Dice": dice.item(),
        "IoU": iou.item(),
        "MAE": mae.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1": f1.item(),
        "Specificity": specificity.item(),
        "Pixel Accuracy": pixel_accuracy.item(),
    }


def validate_net(net, val_loader, criterion, device, writer, epoch):
    """
    验证阶段：计算损失和多种指标
    """
    net.eval()
    val_loss = 0
    metrics = {"Dice": 0, "IoU": 0, "MAE": 0, "Precision": 0, "Recall": 0, "F1": 0, "Specificity": 0, "Pixel Accuracy": 0}
    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            val_loss += loss.item()

            # 计算批次指标
            batch_metrics = compute_metrics(pred, label)
            for key in metrics:
                metrics[key] += batch_metrics[key]

    avg_val_loss = val_loss / len(val_loader)
    for key in metrics:
        metrics[key] /= len(val_loader)

    print(f"Validation Loss: {avg_val_loss}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    for key, value in metrics.items():
        writer.add_scalar(f'Metrics/{key}', value, epoch)

    return avg_val_loss


def train_net(net, device, train_loader, val_loader, epochs, batch_size, lr, weight_decay, momentum, save_path, save_name, log_dir, save_interval, validate_interval, pretrain_path=None):
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    criterion = DiceLoss()
    best_loss = float('inf')

    # 确保保存路径目录存在
    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(log_dir)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss}')

        # 验证阶段
        if (epoch + 1) % validate_interval == 0:
            avg_val_loss = validate_net(net, val_loader, criterion, device, writer, epoch)

            # 保存性能最好的模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_path = os.path.join(save_path, f'{save_name}_best.pth')
                torch.save(net.state_dict(), best_model_path)
                print(f"性能最好的模型已保存到: {best_model_path}")

        # 每隔 save_interval 保存一次模型
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_path, f'{save_name}_epoch_{epoch + 1}.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f"模型检查点已保存到: {checkpoint_path}")

    writer.close()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def prepare_dataloader(config, is_train=True):
    if is_train:
        image_dir = config['data']['image_dir']
        mask_dir = config['data']['mask_dir']
    else:
        image_dir = config['data']['val_image_dir']
        mask_dir = config['data']['val_mask_dir']

    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']
    image_ext = config['data']['image_ext']
    mask_ext = config['data']['mask_ext']

    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def main(config_path='config.yaml'):
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=config['model']['channels'], n_classes=config['model']['n_classes'])
    net.to(device=device)

    train_loader = prepare_dataloader(config, is_train=True)
    val_loader = prepare_dataloader(config, is_train=False)

    save_interval = config['training']['save_interval']
    validate_interval = config['training']['validate_interval']
    pretrain_path = config['training'].get('pretrain_path', None)

    # 调试信息
    print(f"Training parameters: {config['training']}")

    train_net(
        net=net,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        batch_size=config['data']['batch_size'],
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        momentum=float(config['training']['momentum']),
        save_path=config['training']['save_path'],  # 保存路径
        save_name=config['training']['save_name'],  # 保存名称
        log_dir=config['training']['log_dir'],
        save_interval=save_interval,
        validate_interval=validate_interval,
        pretrain_path=pretrain_path
    )


if __name__ == "__main__":
    main()
