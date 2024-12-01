import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入 tqdm

# 定义转换操作
transform = transforms.Compose([
    transforms.ToPILImage(),  # 将 NumPy 数组转为 PIL 图像
    transforms.Resize((256, 256)),  # 调整图像大小为 256x256
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])


# 自定义数据集
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 转换为 RGB 格式（torchvision 需要 RGB 格式）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 确保掩膜值是 0 或 1
        mask = np.where(mask > 127, 1, 0)  # 如果像素值大于 127，则设为 1，其他设为 0

        # 将掩膜转换为 tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        # 调整图像和掩膜大小，确保它们具有相同的尺寸
        if self.transform:
            # 对图像应用变换
            image = self.transform(image)
            mask = cv2.resize(mask.numpy(), (image.shape[1], image.shape[2]))  # 调整掩膜大小
            mask = torch.tensor(mask, dtype=torch.float32)  # 确保掩膜为 tensor 格式

        return image, mask


# 数据集路径
train_image_dir = '../cardiacUDC_TR/four_images'
train_mask_dir = '../cardiacUDC_TR/four_binary'
val_image_dir = '../cardiacUDC_TE/four_images'
val_mask_dir = '../cardiacUDC_TE/four_binary'

# 数据集加载
train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=transform)
val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 定义 UNet++ 模型
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# 损失函数和优化器
loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 自定义评价指标
def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target):
    smooth = 1e-6
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


metrics = {
    'dice': dice_score,
    'iou': iou_score
}

# 训练和验证
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# 评估过程：随机选取10个图片并保存预测结果
def evaluate(model, loader, metrics, epoch, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    losses = []
    scores = {key: [] for key in metrics}

    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            # 使用前10张图片进行评估
            if idx == 1:  # 这里只取一次迭代，即10张图片
                break

            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()  # 确保掩膜的形状是 [batch_size, 1, H, W]

            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            # Resize the output to match the mask size
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            # 计算损失
            loss = loss_fn(outputs, masks)
            losses.append(loss.item())

            # 计算指标
            for key, metric in metrics.items():
                scores[key].append(metric(outputs, masks).item())

            # 保存预测结果
            for i in range(images.size(0)):  # Batch 内的每张图像
                original_image = images[i].cpu().numpy().transpose(1, 2, 0)  # [C, H, W] 转为 [H, W, C]
                original_image = (original_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255  # 恢复 RGB 图像
                original_image = np.clip(original_image, 0, 255).astype(np.uint8)

                prediction = outputs[i].cpu().numpy().squeeze()  # [H, W] 的掩膜
                prediction = (prediction > 0.5).astype(np.uint8) * 255  # 将预测结果转为二值图像

                # 构建保存路径，加入 epoch 编号
                image_filename = loader.dataset.images[idx * loader.batch_size + i]
                save_path = os.path.join(save_dir, f"{epoch+1}_{os.path.splitext(image_filename)[0]}_pred.png")

                # 保存原始图像和预测图像
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(original_image)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("Predicted Mask")
                plt.imshow(prediction, cmap='gray')
                plt.axis('off')

                plt.savefig(save_path)
                plt.close()  # 关闭绘图以节省内存
                print(f"Saved result to {save_path}")

    avg_loss = np.mean(losses)
    avg_scores = {key: np.mean(values) for key, values in scores.items()}
    return avg_loss, avg_scores


# 训练循环
def train():
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            # Resize the output to match the mask size
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 获取10个随机样本进行验证
        indices = np.random.choice(len(val_dataset), 10, replace=False)
        sampler = SubsetRandomSampler(indices)
        val_loader_subset = DataLoader(val_dataset, batch_size=8, sampler=sampler)

        # Validate and save inference results
        val_loss, val_metrics = evaluate(model, val_loader_subset, metrics, epoch, save_dir="outputs")

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        for key, value in val_metrics.items():
            print(f"Val {key}: {value:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "unetplusplus.pth")

# 仅当脚本作为主程序时执行
if __name__ == '__main__':
    train()
