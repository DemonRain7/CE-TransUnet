import os
from PIL import Image, ImageEnhance

# 定义输入文件夹路径和输出文件夹路径
input_folder = '..\\data_input'
output_folder = '..\\data_output'

# 获取输入文件夹中所有的图像文件名
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
               f.endswith('.png') or f.endswith('.jpg')]

# 遍历所有图像文件
for input_path in image_files:
    # 打开图像文件
    img = Image.open(input_path)

    # 对比度增强
    contrast = ImageEnhance.Contrast(img)
    img_enhanced = contrast.enhance(1.5)

    # 定义输出文件名和输出路径
    output_name = os.path.basename(input_path)
    output_path = os.path.join(output_folder, output_name)

    # 保存增强后的图像文件
    img_enhanced.save(output_path)

print("All images processed successfully!")
