import os

# 主文件夹路径
main_folder_path = r'C:\Users\Lenovo\Documents\GitHub\CE-TransUNet\Models\CE-TransUNet\test_results'

highest_accuracy = 0.0
highest_dice = 0.0
highest_iou = 0.0
highest_accuracy_folder = ""
highest_dice_folder = ""
highest_iou_folder = ""

# 用于存储所有记录
all_records = []

# 遍历目录树
for root, dirs, files in os.walk(main_folder_path):
    for dir_name in dirs:
        # 检查文件夹是否以'epoch_'开头
        if dir_name.startswith('epoch_'):
            # 构建log.txt文件路径
            log_file_path = os.path.join(root, dir_name, 'log.txt')

            # 打开txt文件并读取内容
            with open(log_file_path, 'r') as file:
                content = file.readlines()

            # 获取文件夹名称
            folder_name = os.path.basename(os.path.join(root, dir_name))

            # 寻找Average Accuracy、Dice和IoU的行
            accuracy_line = [line for line in content if line.startswith('Average Accuracy:')]
            dice_line = [line for line in content if line.startswith('Average Dice:')]
            iou_line = [line for line in content if line.startswith('Average IoU:')]

            if accuracy_line and dice_line and iou_line:
                # 提取Average Accuracy、Dice和IoU的值
                accuracy_value = float(accuracy_line[0].split(':')[-1].strip()[:-1])
                dice_value = float(dice_line[0].split(':')[-1].strip())
                iou_value = float(iou_line[0].split(':')[-1].strip())

                # 检查是否为最高值
                if accuracy_value > highest_accuracy:
                    highest_accuracy = accuracy_value
                    highest_accuracy_folder = folder_name
                if dice_value > highest_dice:
                    highest_dice = dice_value
                    highest_dice_folder = folder_name
                if iou_value > highest_iou:
                    highest_iou = iou_value
                    highest_iou_folder = folder_name

                # 合并文件夹名称和内容为一行
                combined_line = f"{folder_name}: {', '.join(line.strip() for line in content[-3:])}"

                # 存储所有记录
                all_records.append(combined_line)

# 打印最高的Average Accuracy
highest_accuracy_line = f"Highest Average Accuracy: {highest_accuracy}% in folder {highest_accuracy_folder}"
print(highest_accuracy_line)
all_records.append(highest_accuracy_line)

# 打印最高的Dice
highest_dice_line = f"Highest Dice: {highest_dice} in folder {highest_dice_folder}"
print(highest_dice_line)
all_records.append(highest_dice_line)

# 打印最高的IoU
highest_iou_line = f"Highest IoU: {highest_iou} in folder {highest_iou_folder}"
print(highest_iou_line)
all_records.append(highest_iou_line)

# 打印最高Average Accuracy对应的最后三行
highest_accuracy_log_path = os.path.join(main_folder_path, highest_accuracy_folder, 'log.txt')
with open(highest_accuracy_log_path, 'r') as file:
    highest_accuracy_content = file.readlines()
highest_accuracy_combined_line = f"{highest_accuracy_folder}: {', '.join(line.strip() for line in highest_accuracy_content[-3:])}"
print(highest_accuracy_combined_line)
all_records.append(highest_accuracy_combined_line)

# 打印最高Dice对应的最后三行
highest_dice_log_path = os.path.join(main_folder_path, highest_dice_folder, 'log.txt')
with open(highest_dice_log_path, 'r') as file:
    highest_dice_content = file.readlines()
highest_dice_combined_line = f"{highest_dice_folder}: {', '.join(line.strip() for line in highest_dice_content[-3:])}"
print(highest_dice_combined_line)
all_records.append(highest_dice_combined_line)

# 打印最高IoU对应的最后三行
highest_iou_log_path = os.path.join(main_folder_path, highest_iou_folder, 'log.txt')
with open(highest_iou_log_path, 'r') as file:
    highest_iou_content = file.readlines()
highest_iou_combined_line = f"{highest_iou_folder}: {', '.join(line.strip() for line in highest_iou_content[-3:])}"
print(highest_iou_combined_line)
all_records.append(highest_iou_combined_line)

# 将所有记录保存到文件
output_file_path = os.path.join(main_folder_path, 'all_records.txt')
with open(output_file_path, 'w') as output_file:
    output_file.write('\n'.join(all_records))

print(f"\nAll records have been saved to {output_file_path}")
