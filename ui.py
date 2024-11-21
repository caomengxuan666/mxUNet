# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: ui.py
Author: cmx
Create Date: 2023/2/7
Description：Enhanced UI for dynamic model selection and weight loading
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
from model.Enhanced_Unet import EnhancedUNet
import numpy as np

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('cmx Unet')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        self.output_size = 480
        self.img2predict = ""
        self.origin_shape = ()
        self.model_type = "Old Model"  # 默认模型类型
        self.model = None
        self.mask_path = ""  # 新增，用于存储原始掩膜路径
        self.binary_threshold = 0.5  # 后处理的二值化阈值
        self.initUI()

    def load_model(self, model_type, weights_path=None):
        try:
            if model_type == "Old Model":
                net = UNet(n_channels=3, n_classes=1)
            elif model_type == "New Model":
                net = EnhancedUNet(n_channels=3, n_classes=1)
            else:
                raise ValueError("未识别的模型类型！")

            net.to(device=device)

            # 加载权重
            if weights_path:
                net.load_state_dict(torch.load(weights_path, map_location=device))
            else:
                # 提示用户必须选择权重文件
                QMessageBox.warning(self, "未选择权重文件", "请在右侧先选择权重文件！")
                return None

            net.eval()
            return net
        except Exception as e:
            QMessageBox.critical(self, "模型加载错误", f"无法加载模型: {str(e)}")
            return None

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

        # 模型选择
        model_widget = QWidget()
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型:")
        model_label.setFont(font_main)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Old Model", "New Model"])
        self.model_combo.currentTextChanged.connect(self.change_model)
        load_model_button = QPushButton("加载权重")
        load_model_button.clicked.connect(self.load_custom_weights)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(load_model_button)
        model_widget.setLayout(model_layout)

        # 中间图片展示区
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()

        self.left_img = QLabel("原图")
        self.right_img = QLabel("结果图")
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)

        self.left_img.setMinimumSize(self.output_size, self.output_size)
        self.right_img.setMinimumSize(self.output_size, self.output_size)
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
        up_mask_button = QPushButton("上传掩膜")  # 新增“上传掩膜”按钮
        det_img_button = QPushButton("分割检测")
        compare_mask_button = QPushButton("掩膜对比")  # 新增“掩膜对比”按钮

        up_img_button.clicked.connect(self.upload_img)
        up_mask_button.clicked.connect(self.upload_mask)  # 绑定上传掩膜功能
        det_img_button.clicked.connect(self.detect_img)
        compare_mask_button.clicked.connect(self.compare_masks)  # 绑定掩膜对比功能

        button_layout.addWidget(up_img_button)
        button_layout.addWidget(up_mask_button)
        button_layout.addWidget(det_img_button)

        button_layout.addWidget(compare_mask_button)
        button_widget.setLayout(button_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # 添加掩膜对比选项
        mask_option_widget = QWidget()
        mask_option_layout = QHBoxLayout()

        self.check_draw_contours = QCheckBox("显示分割结果")
        self.check_show_full_mask = QCheckBox("显示原始掩膜")
        # 默认勾选
        self.check_draw_contours.setChecked(True)
        self.check_show_full_mask.setChecked(True)

        mask_option_layout.addWidget(self.check_draw_contours)
        mask_option_layout.addWidget(self.check_show_full_mask)
        mask_option_widget.setLayout(mask_option_layout)

        img_detection_layout.addWidget(img_detection_title)
        img_detection_layout.addWidget(model_widget)
        img_detection_layout.addWidget(mid_img_widget)
        img_detection_layout.addWidget(button_widget)
        img_detection_layout.addWidget(self.progress_bar)
        img_detection_widget.setLayout(img_detection_layout)
        img_detection_layout.addWidget(mask_option_widget)
        self.addTab(img_detection_widget, '图片检测')

    def change_model(self):
        self.model_type = self.model_combo.currentText()
        self.model = self.load_model(self.model_type)

    def load_custom_weights(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择权重文件', '', '*.pth')
        if fileName:
            self.model = self.load_model(self.model_type, weights_path=fileName)
            QMessageBox.information(self, "提示", f"权重 {os.path.basename(fileName)} 加载成功！")

    def upload_img(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择文件', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            save_dir = os.path.join("images", "tmp")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "tmp_upload.jpg")
            shutil.copy(fileName, save_path)

            im0 = cv2.imread(save_path)
            self.origin_shape = im0.shape[:2]

            im_resized = cv2.resize(im0, (self.output_size, self.output_size))
            cv2.imwrite(os.path.join(save_dir, "upload_show_result.jpg"), im_resized)

            self.img2predict = save_path
            self.left_img.setPixmap(QPixmap(os.path.join(save_dir, "upload_show_result.jpg")))
            self.right_img.clear()

    def detect_img(self):
        if not self.img2predict:
            QMessageBox.warning(self, "警告", "请先上传图片！")
            return
        if not self.model:
            QMessageBox.warning(self, "警告", "请先选择并加载模型！")
            return

        save_dir = 'save'
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(self.img2predict).split('.')[0]
        save_res_path = os.path.join(save_dir, f"{base_name}_res.png")

        try:
            img = cv2.imread(self.img2predict)
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))
            img = img / 255.0
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)

            self.progress_bar.setValue(30)

            with torch.no_grad():
                pred = self.model(img_tensor)
                self.progress_bar.setValue(60)
                pred = torch.sigmoid(pred)
                pred = pred.squeeze().detach().cpu().numpy()

            pred[pred >= self.binary_threshold] = 255
            pred[pred < self.binary_threshold] = 0
            pred = pred.astype(np.uint8)

            pred_resized = cv2.resize(pred, (self.origin_shape[1], self.origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(save_res_path, pred_resized)
            self.right_img.setPixmap(QPixmap(save_res_path))
            self.progress_bar.setValue(100)

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.progress_bar.setValue(0)

    def upload_mask(self):
        """
        上传原始掩膜
        """
        fileName, _ = QFileDialog.getOpenFileName(self, '选择掩膜文件', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            self.mask_path = fileName
            QMessageBox.information(self, "提示", f"掩膜 {os.path.basename(fileName)} 上传成功！")


    def compare_masks(self):
        if not self.img2predict:
            QMessageBox.warning(self, "警告", "请先上传图片！")
            return
        if not self.mask_path:
            QMessageBox.warning(self, "警告", "请先上传掩膜！")
            return

        save_dir = 'save'
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(self.img2predict).split('.')[0]
        comparison_path = os.path.join(save_dir, f"{base_name}_comparison.png")

        try:
            self.progress_bar.setValue(10)  # 初始化进度条

            # 读取原始图片
            original_img = cv2.imread(self.img2predict)
            self.progress_bar.setValue(30)  # 更新进度条

            # 读取模型分割结果
            segmented_img_path = os.path.join(save_dir, f"{base_name}_res.png")
            segmented_img = cv2.imread(segmented_img_path, cv2.IMREAD_GRAYSCALE)
            if segmented_img is None:
                raise FileNotFoundError("模型分割结果未生成，请先执行分割检测！")
            segmented_resized = cv2.resize(segmented_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.progress_bar.setValue(50)  # 更新进度条

            # 读取原始掩膜
            mask_img = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.progress_bar.setValue(70)  # 更新进度条

            result_img = original_img.copy()

            # 如果勾选了“绘制轮廓”
            if self.check_draw_contours.isChecked():
                contours_model, _ = cv2.findContours(segmented_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result_img, contours_model, -1, (0, 255, 0), 2)  # 绘制绿色轮廓

            # 如果勾选了“显示整体轮廓”
            if self.check_show_full_mask.isChecked():
                mask_overlay = np.zeros_like(original_img)
                mask_overlay[mask_resized > 0] = (0, 0, 200)  # 填充深红色
                result_img = cv2.addWeighted(result_img, 1, mask_overlay, 0.5, 0)

            self.progress_bar.setValue(90)  # 更新进度条

            # 保存并显示对比结果
            cv2.imwrite(comparison_path, result_img)
            self.right_img.setPixmap(QPixmap(comparison_path))

            self.progress_bar.setValue(100)  # 完成

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.progress_bar.setValue(0)  # 重置进度条



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
