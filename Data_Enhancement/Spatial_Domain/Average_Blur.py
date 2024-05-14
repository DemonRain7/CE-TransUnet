import cv2
import os
import json

# 读取 JSON 文件并解析内容
with open('../config.json', 'r') as f:
    config = json.load(f)

# 从解析后的 JSON 内容中获取所需信息
input_path = config['input_path']
output_path = config['output_path']


# 读取文件夹中的所有图像
for filename in os.listdir(input_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 读取图像
        img = cv2.imread(os.path.join(input_path, filename))
        # 进行均值滤波
        img = cv2.blur(img, (5,5))
        # 保存新的图像到输出文件夹
        cv2.imwrite(os.path.join(output_path, filename), img)
