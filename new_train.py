import yaml
from torch.utils.tensorboard import SummaryWriter
from model.Enhanced_Unet import EnhancedUNet  # 修改为引入 EnhancedUNet
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, config_path="newconfig.yaml"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 读取配置
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

        if self.channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1e-7) / (inputs.sum() + targets.sum() + 1e-7)
        return bce + (1 - dice)


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

            # 模型前向传播
            pred = net(image)

            # 调整目标掩码尺寸以匹配预测输出
            label = F.interpolate(label, size=pred.shape[2:], mode="bilinear", align_corners=False)

            # 计算损失
            loss = criterion(pred, label)
            val_loss += loss.item()

            # 计算指标
            batch_metrics = compute_metrics(pred, label)
            for key in metrics:
                metrics[key] += batch_metrics[key]

    # 计算平均损失和指标
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



def train_net(net, device, train_loader, val_loader, args):
    optimizer = optim.Adam(net.parameters(), lr=float(args["learning_rate"]), weight_decay=float(args["weight_decay"]))
    criterion = BCEDiceLoss()
    best_loss = float('inf')

    # 确保保存路径存在
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

                # 调整目标尺寸与模型输出一致
                pred = net(image)
                label = F.interpolate(label, size=pred.shape[2:], mode="bilinear", align_corners=False)

                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        print(f'Epoch {epoch + 1}/{args["epochs"]} - Loss: {avg_epoch_loss}')

        # 验证阶段
        if (epoch + 1) % args["validate_interval"] == 0:
            avg_val_loss = validate_net(net, val_loader, criterion, device, writer, epoch)

            # 保存性能最好的模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss

                # 动态构造最优模型保存路径
                best_model_path = os.path.join(args["save_path"], f'{args["save_name"]}_best.pth')
                torch.save(net.state_dict(), best_model_path)
                print(f"性能最好的模型已保存到: {best_model_path}")

        # 每隔 save_interval 保存一次模型
        if (epoch + 1) % args["save_interval"] == 0:
            # 动态构造检查点模型保存路径
            checkpoint_path = os.path.join(args["save_path"], f'{args["save_name"]}_epoch_{epoch + 1}.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f"模型检查点已保存到: {checkpoint_path}")

    writer.close()


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
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def main(config_path='newconfig.yaml'):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = EnhancedUNet(n_channels=config['model']['channels'], n_classes=config['model']['n_classes'])
    net.to(device=device)

    train_loader = prepare_dataloader(config, is_train=True)
    val_loader = prepare_dataloader(config, is_train=False)
    print(f"Training parameters: {config['training']}")


    train_net(
        net=net,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        args=config['training']
    )


if __name__ == "__main__":
    main()
