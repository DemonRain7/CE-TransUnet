import cv2
import os
import json

# 读取 JSON 配置文件
with open('../config.json', 'r') as f:
    config = json.load(f)

# 从配置文件中获取所需信息
input_folder = config['input_path']
output_folder = config['output_path']
sigma_color = config['sigma_color']
sigma_space = config['sigma_space']

# 遍历输入文件夹中的所有图像
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 读取图像
        img = cv2.imread(os.path.join(input_folder, filename))
        print('Reading image:', filename)

        # 进行双边滤波处理
        img_filtered = cv2.bilateralFilter(img, 0, sigma_color, sigma_space)
        print('Filtering image:', filename)

        # 保存新的图像到输出文件夹
        cv2.imwrite(os.path.join(output_folder, filename), img_filtered)
        print('Saving image:', filename)
