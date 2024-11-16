import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    net = UNet(n_channels=3, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('outputModels/best_model.pth', map_location=device))
    net.eval()

    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    images_dir = r'D:\medicalDetect\ISIC2018_Task1_Training_'
    images_pattern = os.path.join(images_dir, '*.jpg')
    tests_path = glob.glob(images_pattern)
    print(f"Found {len(tests_path)} images with pattern {images_pattern}")

    if not tests_path:
        print("No images found. Please check the path and try again.")
    else:
        save_dir = os.path.join(images_dir, 'save')
        os.makedirs(save_dir, exist_ok=True)

        for test_path in tests_path:
            save_res_path = os.path.join(save_dir, os.path.basename(test_path).split('.')[0] + '_res.png')
            img = cv2.imread(test_path)
            if img is None:
                print(f"Failed to read image: {test_path}")
                continue

            origin_shape = img.shape
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))  # 转换为C*H*W
            img = img / 255.0  # 归一化
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = net(img_tensor)
                pred = torch.sigmoid(pred)  # 确保在推理时应用sigmoid
                pred = pred.squeeze().cpu().numpy()
                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0
                pred = pred.astype(np.uint8)

            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(save_res_path, pred)
            print(f"Processed image saved to {save_res_path}")
