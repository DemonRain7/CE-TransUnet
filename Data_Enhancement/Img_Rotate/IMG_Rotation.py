import json
from PIL import Image
import os
import random

with open('../config.json', 'r') as f :
    config = json.load(f)

folder_path = config['input_path']
rotation_angles = [90, 180, 270]
rotated_path = config['rotate_path']


# 获取文件夹中所有图片文件的路径
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

for image_path in image_paths:
    # 打开图片
    with Image.open(image_path) as img:
        # 随机选择一个角度进行翻转
        rotation_angle = random.choice(rotation_angles)
        # 顺时针旋转指定角度
        rotated_img = img.rotate(rotation_angle)
        # 保存翻转后的图片，覆盖原来的图片
        rotated_img.save(os.path.join(rotated_path, os.path.basename(image_path)), format='JPEG')
