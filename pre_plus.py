import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from unetPlus import model

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageInferenceApp(QWidget):
    def __init__(self, model_type="UNet++", weights_path=None):
        super().__init__()

        self.model_type = model_type
        self.model = None
        self.weights_path = weights_path
        self.load_model(weights_path)

        # 新增原图显示的QLabel
        self.original_image_label = QLabel(self)
        self.original_image_label.setAlignment(Qt.AlignCenter)

        self.init_ui()


    def init_ui(self):
        self.setWindowTitle('Image Inference')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        # Add widgets
        self.select_image_button = QPushButton('Select Image', self)
        self.select_image_button.clicked.connect(self.select_image)

        self.layout.addWidget(self.select_image_button)

        # 添加原图显示的QLabel
        self.layout.addWidget(self.original_image_label)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.image_label)

        self.inference_button = QPushButton('Start Inference', self)
        self.inference_button.clicked.connect(self.start_inference)

        self.layout.addWidget(self.inference_button)

        self.setLayout(self.layout)

        self.selected_image_path = None
        self.original_pixmap = None

    def display_image(self, img_path):
        """显示选择的图片"""
        image = QImage(img_path)
        self.original_pixmap = QPixmap(image)
        self.original_image_label.setPixmap(self.original_pixmap.scaled(self.original_image_label.size(), Qt.KeepAspectRatio))

    def start_inference(self):
        """启动推理"""
        if not self.selected_image_path:
            return

        # 推理过程
        pred = self.evaluate(self.selected_image_path)

        if pred is not None:
            self.show_inference_result(pred)
    def show_inference_result(self, pred):
        """显示推理结果"""
        # 显示原图
        self.original_image_label.setPixmap(self.original_pixmap.scaled(self.original_image_label.size(), Qt.KeepAspectRatio))

        # 显示预测结果图像
        pred_image = QImage(pred.data, pred.shape[1], pred.shape[0], pred.shape[1], QImage.Format_Grayscale8)
        pred_pixmap = QPixmap(pred_image)
        self.image_label.setPixmap(pred_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

        # 保存结果
        save_path = os.path.join(os.path.dirname(self.selected_image_path), "predicted_mask.png")
        cv2.imwrite(save_path, pred)
        print(f"预测结果已保存到: {save_path}")

    def evaluate(self, img_path):
        """
        与evaluate方法类似，用于推理
        """
        if not self.model:
            raise RuntimeError("模型未加载，请确保模型和权重已正确加载。")

        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像文件: {img_path}")

            # 调整图像大小并预处理
            img_resized = cv2.resize(img, (256, 256))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device=device)

            # 推理
            with torch.no_grad():
                outputs = self.model(img_tensor)

            # 后处理
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze().cpu().numpy()

            # 应用二值化
            outputs[outputs >= 0.5] = 255
            outputs[outputs < 0.5] = 0
            outputs = outputs.astype(np.uint8)

            return outputs

        except Exception as e:
            print(f"推理时出错: {str(e)}")
            return None
    def load_model(self, weights_path):
        """
        加载UNet++模型和权重
        """
        try:
            if self.model_type == "UNet++":
                self.model = model
            else:
                raise ValueError("未识别的模型类型!")

            self.model.to(device=device)

            # 加载权重
            if weights_path:
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
            else:
                raise ValueError("权重文件路径为空，请提供正确的路径！")

            self.model.eval()  # 设置为评估模式

        except Exception as e:
            print(f"加载模型时出错: {str(e)}")

    def select_image(self):
        """选择图片"""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.selected_image_path = file_path
            self.display_image(file_path)

    def show_inference_result(self, pred):
        """显示推理结果"""
        # 显示原图
        self.original_image_label.setPixmap(self.original_pixmap.scaled(self.original_image_label.size(), Qt.KeepAspectRatio))

        # 显示预测结果图像
        pred_image = QImage(pred.data, pred.shape[1], pred.shape[0], pred.shape[1], QImage.Format_Grayscale8)
        pred_pixmap = QPixmap(pred_image)
        self.image_label.setPixmap(pred_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

        # 保存结果
        save_path = os.path.join(os.path.dirname(self.selected_image_path), "predicted_mask.png")
        cv2.imwrite(save_path, pred)
        print(f"预测结果已保存到: {save_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 模型和权重文件路径
    weights_path = "unetplusplus.pth"

    # 创建应用窗口
    window = ImageInferenceApp(weights_path=weights_path)
    window.show()

    sys.exit(app.exec_())
