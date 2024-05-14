import os
import cv2
import numpy as np
import json

# 读取配置文件
with open("../config.json", "r") as f:
    config = json.load(f)

# 输入文件夹路径
input_folder = config["input_path"]
# 输出文件夹路径
output_folder = config["output_path"]


# 高斯低通滤波器函数
def gaussian_lp_filter(img, d):
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    # 创建一个与图像大小相同的网格
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    # 创建一个高斯低通滤波器
    lp_filter = np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * d ** 2))
    # 移动低频分量到图像中心
    lp_filter = np.fft.fftshift(lp_filter)
    # 将滤波器与图像进行卷积
    filtered_img = np.fft.ifft2(np.fft.fft2(img) * lp_filter)
    # 获取滤波结果的实部
    filtered_img = np.real(filtered_img)
    # 将像素值缩放到0-255范围内，并转换为8位整型
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return filtered_img


# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    # 检查文件扩展名是否为图像格式（可以根据需要添加或修改）
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # 读取图像
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        # 对图像进行高斯低通滤波
        filtered_img = gaussian_lp_filter(img, 30)
        # 保存处理后的图像到输出文件夹
        cv2.imwrite(os.path.join(output_folder, filename), filtered_img)
