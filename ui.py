# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: ui.py
Author: cmx
Create Date: 2023/2/7
Description：
-------------------------------------------------
"""
import os
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from model.unet_model import UNet
import numpy as np

# 窗口主类
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('基于UNet的皮肤疾病病灶区域分割')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        self.output_size = 480
        self.img2predict = ""
        self.origin_shape = ()
        self.model = self.load_model()
        self.initUI()

    def load_model(self):
        net = UNet(n_channels=3, n_classes=1)
        net.to(device=device)
        net.load_state_dict(torch.load('best_model.pth', map_location=device))
        net.eval()
        return net

    def initUI(self):
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)

        # 图片检测子界面
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()

        # 标题
        img_detection_title = QLabel("病理性检测模块")
        img_detection_title.setFont(font_title)
        img_detection_title.setAlignment(Qt.AlignCenter)

        # 中间图片展示区
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()

        self.left_img = QLabel("原图")
        self.right_img = QLabel("结果图")
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)

        # 设置固定的最小大小，确保两张图片的显示尺寸相同
        self.left_img.setMinimumSize(self.output_size, self.output_size)
        self.right_img.setMinimumSize(self.output_size, self.output_size)

        # 使图像自适应并且保证尺寸一致
        self.left_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.left_img.setScaledContents(True)
        self.right_img.setScaledContents(True)

        mid_img_layout.addWidget(self.left_img, stretch=1)
        mid_img_layout.addWidget(self.right_img, stretch=1)
        mid_img_widget.setLayout(mid_img_layout)

        # 按钮区
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        merge_img_button = QPushButton("合并图像")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        merge_img_button.clicked.connect(self.merge_images)
        button_layout.addWidget(up_img_button)
        button_layout.addWidget(det_img_button)
        button_layout.addWidget(merge_img_button)
        button_widget.setLayout(button_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        img_detection_layout.addWidget(img_detection_title)
        img_detection_layout.addWidget(mid_img_widget)
        img_detection_layout.addWidget(button_widget)
        img_detection_layout.addWidget(self.progress_bar)
        img_detection_widget.setLayout(img_detection_layout)

        self.addTab(img_detection_widget, '图片检测')

    def upload_img(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择文件', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            save_dir = os.path.join("images", "tmp")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "tmp_upload.jpg")
            shutil.copy(fileName, save_path)

            # 读取图像并保存原始形状
            im0 = cv2.imread(save_path)
            self.origin_shape = im0.shape[:2]

            # 调整图像尺寸到统一大小
            im_resized = cv2.resize(im0, (self.output_size, self.output_size))
            cv2.imwrite(os.path.join(save_dir, "upload_show_result.jpg"), im_resized)

            self.img2predict = save_path
            self.left_img.setPixmap(QPixmap(os.path.join(save_dir, "upload_show_result.jpg")))
            self.right_img.clear()

    def detect_img(self):
        if not self.img2predict:
            QMessageBox.warning(self, "警告", "请先上传图片！")
            return

        save_dir = 'save'
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(self.img2predict).split('.')[0]
        save_res_path = os.path.join(save_dir, f"{base_name}_res.png")

        try:
            img = cv2.imread(self.img2predict)
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))  # 转换为C*H*W
            img = img / 255.0  # 归一化
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)

            self.progress_bar.setValue(30)

            with torch.no_grad():
                pred = self.model(img_tensor)
                self.progress_bar.setValue(60)
                pred = torch.sigmoid(pred)
                pred = pred.squeeze().detach().cpu().numpy()

            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = pred.astype(np.uint8)

            # 恢复到原始大小
            pred_resized = cv2.resize(pred, (self.origin_shape[1], self.origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(save_res_path, pred_resized)
            self.right_img.setPixmap(QPixmap(save_res_path))
            self.progress_bar.setValue(100)

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.progress_bar.setValue(0)

    def merge_images(self):
        if not self.img2predict:
            QMessageBox.warning(self, "警告", "请先上传图片并进行检测！")
            return

        save_dir = 'save'
        base_name = os.path.basename(self.img2predict).split('.')[0]
        original_img_path = self.img2predict
        segmented_img_path = os.path.join(save_dir, f"{base_name}_res.png")

        try:
            original_img = cv2.imread(original_img_path)
            segmented_img = cv2.imread(segmented_img_path, cv2.IMREAD_GRAYSCALE)

            segmented_img_resized = cv2.resize(segmented_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(segmented_img_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original_img, contours, -1, (0, 255, 0), 2)

            merged_img_path = os.path.join(save_dir, f"{base_name}_merged.png")
            cv2.imwrite(merged_img_path, original_img)
            self.right_img.setPixmap(QPixmap(merged_img_path))

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
