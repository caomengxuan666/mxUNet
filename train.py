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
    def __init__(self, image_dir, mask_dir, transform=None,config_path="config.yaml"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.config = load_config(config_path)
        self.channels = self.config['model']['channels']
        print("当前训练的 图像通道数:  ", self.channels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        #mask_name = img_name.replace('.jpg', '.png')

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


def train_net(net, device, train_loader, epochs, batch_size, lr, weight_decay, momentum, save_path, log_dir, save_interval, pretrain_path=None):
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    criterion = DiceLoss()
    best_loss = float('inf')

    # 如果有预训练模型路径，加载预训练模型权重
    if pretrain_path:
        if os.path.exists(pretrain_path):
            net.load_state_dict(torch.load(pretrain_path))
            print(f"加载预训练模型：{pretrain_path}")
        else:
            print(f"警告：预训练模型 {pretrain_path} 未找到，开始从头训练。")

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

        # 每隔 save_interval 保存一次模型，并添加数字
        if (epoch + 1) % save_interval == 0:
            save_model_path = f"{save_path}_epoch{epoch + 1}.pth"
            torch.save(net.state_dict(), save_model_path)
            print(f"模型已保存: {save_model_path}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(net.state_dict(), save_path)

    writer.close()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def prepare_dataloader(config):
    image_dir = config['data']['image_dir']
    mask_dir = config['data']['mask_dir']
    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']

    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(config_path='config.yaml'):
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=config['model']['channels'], n_classes=config['model']['n_classes'])
    net.to(device=device)

    train_loader = prepare_dataloader(config)

    save_interval = config['training']['save_interval']

    pretrain_path = config['training'].get('pretrain_path', None)

    # 调试信息
    print(f"Training parameters: {config['training']}")

    train_net(
        net=net,
        device=device,
        train_loader=train_loader,
        epochs=config['training']['epochs'],
        batch_size=config['data']['batch_size'],
        lr=float(config['training']['learning_rate']),  # 确保转换为浮点数
        weight_decay=float(config['training']['weight_decay']),  # 确保转换为浮点数
        momentum=float(config['training']['momentum']),  # 确保转换为浮点数
        save_path=config['training']['save_path'],
        log_dir=config['training']['log_dir'],
        save_interval=save_interval,
        pretrain_path=pretrain_path
    )



if __name__ == "__main__":
    main()
