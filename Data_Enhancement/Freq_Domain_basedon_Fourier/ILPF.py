import cv2
import os
import numpy as np
import json

# 在上述代码中，我们首先使用 json.load() 函数读取 config.json 文件并解析其内容。然后，我们从解析后的 JSON 内容中获取所需信息 input_folder、output_folder 和 cutoff_frequency。
# 接下来，我们定义了一个理想低通滤波器函数 ideal_lp_filter()。该函数可以根据输入的图像尺寸和截止频率生成一个相应的理想低通滤波器。在本例中，我们使用了矩形频谱，也就是在频域上以半径为 cutoff_frequency 的圆形区域内保留所有频率，而在圆形区域外则将所有频率设为零。

# 读取 JSON 配置文件
with open('../config.json', 'r') as f:
    config = json.load(f)

# 从配置文件中获取所需信息
input_folder = config['input_path']
output_folder = config['output_path']
cutoff_frequency = config['cutoff_frequency']


# 定义理想低通滤波器函数
def ideal_lp_filter(size, cutoff):
    rows, cols = size
    x = np.linspace(-0.5, 0.5, cols) * cols
    y = np.linspace(-0.5, 0.5, rows) * rows
    radius = np.sqrt((x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis])
    h = np.zeros((rows, cols), np.float32)
    h[radius <= cutoff] = 1
    return h


# 遍历输入文件夹中的所有图像
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 读取图像
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

        # 进行理想低通滤波处理
        rows, cols = img.shape
        h = ideal_lp_filter((rows, cols), cutoff_frequency)
        img_filtered = np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fftshift(h)))

        # 将图像数值归一化到 [0, 255] 范围内
        img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # 保存新的图像到输出文件夹
        cv2.imwrite(os.path.join(output_folder, filename), img_filtered)
