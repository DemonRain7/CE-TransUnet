from PIL import Image
import os

# 输入文件夹路径和输出文件夹路径
input_folder = r"data/mask_input"
output_folder = r"data/SegmentationClass"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中所有PNG图片的文件列表
png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

for png_file in png_files:
    # 构建输入文件的完整路径
    input_path = os.path.join(input_folder, png_file)

    # 打开图像文件
    img = Image.open(input_path)

    # 将像素值为255的像素替换为1
    img = img.point(lambda p: p if p != 255 else 1)

    # 构建输出文件的完整路径
    output_path = os.path.join(output_folder, png_file)

    # 保存处理后的图像到输出文件夹中
    img.save(output_path)

    print(f"Processed: {input_path} -> {output_path}")

print("Processing completed.")
