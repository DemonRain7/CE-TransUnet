import cv2
import numpy as np
import os
import json

# 读取配置文件
with open('../config.json', 'r') as f:
    config = json.load(f)

# 输入文件夹路径
input_folder = config['input_path']
# 输出文件夹路径
output_folder = config['output_path']


# 定义布特沃斯低通滤波器函数
def butterworth_lp_filter(shape, cutoff, n):
    """
    生成一个布特沃斯低通滤波器。
    :param shape: 图像的尺寸，例如 (512, 512)。
    :param cutoff: 滤波器的截止频率。
    :param n: 滤波器的斜率。
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x - ccol
    y = y - crow
    d = np.sqrt(x ** 2 + y ** 2)
    return 1 / (1 + (d / cutoff) ** (2 * n))


# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    # 读取灰度图像
    img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

    # 进行布特沃斯低通滤波
    lp_filter = butterworth_lp_filter(img.shape, cutoff=50, n=2)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift = fshift * lp_filter
    f_ishift = np.fft.ifftshift(fshift)
    img_lp = np.abs(np.fft.ifft2(f_ishift))

    # 保存滤波后的图像
    cv2.imwrite(os.path.join(output_folder, 'lp_filtered_' + filename), img_lp)
