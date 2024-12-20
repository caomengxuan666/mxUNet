import os
import shutil
import sys
import cv2
import torch
import os.path as osp
from model.unet_model import UNet
from model.Enhanced_Unet import EnhancedUNet
import numpy as np
from simpleTTK import ReadImage
def inference_image(model, image_path, output_size, binary_threshold, origin_shape, device):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (output_size, output_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = img / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            pred = pred.squeeze().detach().cpu().numpy()

        pred[pred >= binary_threshold] = 255
        pred[pred < binary_threshold] = 0
        pred = pred.astype(np.uint8)

        pred_resized = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)

        return pred_resized

    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def compute_dice(img1,img2):
    insersection=img1*img2

    dice=(2*intersection.sum())/(img1.size()+img2.size())
    return dice

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred=inference_image("/home/chxy/mxUNet/outputModels/b4_BCED1130_best.pth","/home/chxy/cardiacUDC_TE/four_images/0.png",0.5
                    ,[256,256],device)
    compute_dice(pred,ReadImage("/home/chxy/cardiacUDC_TE/four_images/0.png"))