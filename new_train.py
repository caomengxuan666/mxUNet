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
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from losses import BCEDiceLoss,FocalBCEDiceLoss,BCETverskyLoss,DiceLoss,WeightedDiceLoss

# 加载配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, config_path="newconfig.yaml"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

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
        try:
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            mask_name = img_name.replace(self.image_ext, self.mask_ext)
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

            # 加载图像和掩膜
            image = Image.open(img_path).convert("RGB" if self.channels != 1 else "L")
            mask = Image.open(mask_path).convert("L")  # 确保掩膜是单通道

            # 应用同步的几何变换
            if self.image_transform or self.mask_transform:
                # 随机水平翻转
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)

                # 随机垂直翻转
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)

                # 随机旋转
                angles = [0, 90, 180, 270]
                angle = random.choice(angles)
                if angle != 0:
                    image = TF.rotate(image, angle)
                    mask = TF.rotate(mask, angle)

                # 随机平移
                if random.random() > 0.5:
                    shift_x = random.randint(-10, 10)
                    shift_y = random.randint(-10, 10)
                    image = TF.affine(image, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
                    mask = TF.affine(mask, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)

                # 随机缩放
                if random.random() > 0.5:
                    scale_factor = random.uniform(0.8, 1.2)
                    image = TF.resize(image, (int(image.size[1] * scale_factor), int(image.size[0] * scale_factor)))
                    mask = TF.resize(mask, (int(mask.size[1] * scale_factor), int(mask.size[0] * scale_factor)))

                # 随机裁剪和缩放
                if self.image_transform and self.mask_transform:
                    i, j, h, w = transforms.RandomResizedCrop.get_params(
                        image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                    )
                    # 使用配置中的图像大小
                    target_size = self.config['data']['image_size']
                    # 确保 target_size 是一个元组
                    if isinstance(target_size, list):
                        target_size = tuple(target_size)
                    elif isinstance(target_size, int):
                        target_size = (target_size, target_size)
                    else:
                        raise ValueError(f"Unsupported image_size format: {target_size}")
                    image = TF.resized_crop(image, i, j, h, w, size=target_size, interpolation=Image.BILINEAR)
                    mask = TF.resized_crop(mask, i, j, h, w, size=target_size, interpolation=Image.NEAREST)

            # 应用转换
            if self.image_transform:
                image = self.image_transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)

            # 确保掩膜是二值化且为 Float 类型
            mask = (mask > 0).float()

            # 确保图像和掩膜的空间尺寸一致（忽略通道数）
            assert image.shape[1:] == mask.shape[1:], f"Image and mask spatial shapes do not match: {image.shape[1:]} vs {mask.shape[1:]}"

            return image, mask

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            # 这里可以选择跳过有问题的数据或重新抛出异常
            raise e

# 计算多个指标
def compute_metrics(pred, label):
    pred = torch.sigmoid(pred) > 0.5  # 二值化
    pred = pred.view(-1).float()
    label = label.view(-1).float()

    intersection = (pred * label).sum()
    union = pred.sum() + label.sum() - intersection
    TP = intersection
    FP = pred.sum() - intersection
    FN = label.sum() - intersection
    TN = (1 - pred).sum() - FN

    dice = (2. * TP) / (2. * TP + FP + FN + 1e-7)
    iou = TP / (union + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    specificity = TN / (TN + FP + 1e-7)
    pixel_accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-7)
    mae = torch.abs(pred - label).mean()

    return {
        "Dice": dice.item(),
        "IoU": iou.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1": f1.item(),
        "Specificity": specificity.item(),
        "Pixel Accuracy": pixel_accuracy.item(),
        "MAE": mae.item(),
    }


# 数据增强（训练集）仅对图像应用颜色相关变换
def get_train_transform(image_size):
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])


# 数据增强（验证集）仅进行必要的变换
def get_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


# 数据加载器准备函数
def prepare_dataloader(config, is_train=True):
    image_size = config['data']['image_size']
    # 确保 image_size 是一个元组
    if isinstance(image_size, list):
        image_size = tuple(image_size)
    elif isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        raise ValueError(f"Unsupported image_size format: {image_size}")

    if is_train:
        image_dir = config['data']['image_dir']
        mask_dir = config['data']['mask_dir']
        image_transform = get_train_transform(image_size)
        mask_transform = get_val_transform(image_size)  # 仅转换为张量和调整大小
    else:
        image_dir = config['data']['val_image_dir']
        mask_dir = config['data']['val_mask_dir']
        image_transform = get_val_transform(image_size)
        mask_transform = get_val_transform(image_size)

    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    return DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=is_train,
        num_workers=4,  # 根据CPU核数调整
        pin_memory=True if torch.cuda.is_available() else False
    )


# 验证阶段
def validate_net(net, val_loader, criterion, device, writer, epoch, config, save_samples=True, save_path=""):
    net.eval()
    val_loss = 0
    metrics = {key: 0 for key in ["Dice", "IoU", "Precision", "Recall", "F1", "Specificity", "Pixel Accuracy", "MAE"]}
    sample_images = []
    sample_preds = []
    sample_labels = []

    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(image)
            label = F.interpolate(label, size=pred.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(pred, label)
            val_loss += loss.item()

            # 计算指标
            batch_metrics = compute_metrics(pred, label)
            for key in metrics:
                metrics[key] += batch_metrics[key]

            # 保存部分样本用于可视化
            if save_samples and len(sample_images) < 5:
                batch_size = image.size(0)
                for j in range(batch_size):
                    if len(sample_images) >= 5:
                        break
                    sample_images.append(image[j].cpu())
                    sample_preds.append(torch.sigmoid(pred[j]).cpu())
                    sample_labels.append(label[j].cpu())

    # 可视化样本预测结果
    if save_samples and save_path:
        # 获取 save_path 的上一级目录
        parent_dir = os.path.dirname(save_path)
        # 构建目标路径
        target_dir = os.path.join(parent_dir, 'res')
        os.makedirs(target_dir, exist_ok=True)  # 确保目标目录存在
        for i in range(len(sample_images)):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            # 转换为适合显示的格式
            if config['model']['channels'] == 1:
                axs[0].imshow(sample_images[i].squeeze(), cmap='gray')
            else:
                # 确保图像在 [0,1] 范围内
                image_np = sample_images[i].permute(1, 2, 0).numpy()
                image_np = np.clip(image_np, 0, 1)
                axs[0].imshow(image_np)
            axs[0].set_title('Image')
            axs[0].axis('off')
            axs[1].imshow(sample_labels[i].squeeze(), cmap='gray')
            axs[1].set_title('Mask')
            axs[1].axis('off')
            axs[2].imshow(sample_preds[i].squeeze(), cmap='gray')
            axs[2].set_title('Prediction')
            axs[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, f'validation_sample_epoch_{epoch + 1}_{i+1}.png'))
            plt.close()

    # 计算平均损失和指标
    avg_val_loss = val_loss / len(val_loader)
    for key in metrics:
        metrics[key] /= len(val_loader)

    print(f"Validation Loss: {avg_val_loss}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    for key, value in metrics.items():
        writer.add_scalar(f'Metrics/val_{key}', value, epoch)

    return avg_val_loss


# 训练函数
def train_net(net, device, train_loader, val_loader, args, config):
    optimizer = optim.Adam(net.parameters(), lr=float(args["learning_rate"]), weight_decay=float(args["weight_decay"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = BCEDiceLoss()
    best_loss = float('inf')
    patience = 25  # 允许验证损失不下降的最大epoch数
    trigger_times = 0

    os.makedirs(args["save_path"], exist_ok=True)
    writer = SummaryWriter(args["log_dir"])

    for epoch in range(args["epochs"]):
        net.train()
        epoch_loss = 0
        metrics = {key: 0 for key in ["Dice", "IoU", "Precision", "Recall", "F1", "Specificity", "Pixel Accuracy", "MAE"]}

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

                # 计算指标
                batch_metrics = compute_metrics(pred, label)
                for key in metrics:
                    metrics[key] += batch_metrics[key]

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

        avg_epoch_loss = epoch_loss / len(train_loader)
        for key in metrics:
            metrics[key] /= len(train_loader)

        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        for key, value in metrics.items():
            writer.add_scalar(f'Metrics/train_{key}', value, epoch)

        print(f"Epoch {epoch + 1}/{args['epochs']} - Loss: {avg_epoch_loss}")
        for key, value in metrics.items():
            print(f"Train {key}: {value:.4f}")

        # 验证
        if (epoch + 1) % args["validate_interval"] == 0:
            avg_val_loss = validate_net(
                net, val_loader, criterion, device, writer, epoch,
                config=config,
                save_samples=True,
                save_path=args["save_path"]
            )
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_path = os.path.join(args["save_path"], f'{args["save_name"]}_best.pth')
                torch.save(net.state_dict(), best_model_path)
                print(f"性能最好的模型已保存到: {best_model_path}")
                trigger_times = 0
            else:
                trigger_times += 1
                print(f'EarlyStopping counter: {trigger_times} out of {patience}')
                if trigger_times >= patience:
                    print('Early stopping!')
                    break

        # 保存检查点
        if (epoch + 1) % args["save_interval"] == 0:
            checkpoint_path = os.path.join(args["save_path"], f'{args["save_name"]}_epoch_{epoch + 1}.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f"模型检查点已保存到: {checkpoint_path}")

    writer.close()


# 主函数
def main(config_path='newconfig.yaml'):
    # 加载配置
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 打印完整配置内容
    print("Loaded configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # 初始化模型
    net = EnhancedUNet(n_channels=config['model']['channels'], n_classes=config['model']['n_classes'])
    net.to(device=device)

    # 准备数据加载器
    train_loader = prepare_dataloader(config, is_train=True)
    val_loader = prepare_dataloader(config, is_train=False)

    # 检查数据加载器
    try:
        images, masks = next(iter(train_loader))
        print(f"Train batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
    except Exception as e:
        print(f"Error loading train batch: {e}")
        return

    try:
        images, masks = next(iter(val_loader))
        print(f"Validation batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
    except Exception as e:
        print(f"Error loading validation batch: {e}")
        return

    # 开始训练
    train_net(
        net=net,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        args=config['training'],
        config=config  # 传递config给train_net
    )

    # 测试模型结构（可选）
    # model = EnhancedUNet(n_channels=3, n_classes=1)
    # model.eval()
    # input_tensor = torch.randn(1, 3, 256, 256)
    # output = model(input_tensor)
    # print(output.shape)  # 应该输出 torch.Size([1, 1, 256, 256]))


if __name__ == "__main__":

    main()
