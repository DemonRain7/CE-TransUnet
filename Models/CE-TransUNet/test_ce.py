import sys
import torch
from torch.utils.data import DataLoader
from data import MyDataset
from ce_net import *
import os
from PIL import Image
import re
import matplotlib.pyplot as plt
from ce_net import *
from atUnet import *
from new_unet import *
from LeUNet import *
from ssformer import *
from MTUNet import *
from new_unet import *

def compute_metrics(predictions, labels):
    with torch.no_grad():
        predicted_labels = torch.argmax(predictions, dim=1)

        # Compute accuracy
        correct_predictions = (predicted_labels == labels).float()
        accuracy = correct_predictions.mean()

        # Compute Dice coefficient
        intersection = torch.sum(predicted_labels * labels)
        union = torch.sum(predicted_labels) + torch.sum(labels)
        dice_coefficient = (2.0 * intersection) / (union + 1e-7)

        # Compute IoU (Intersection over Union)
        iou = intersection / (union - intersection + 1e-7)

    return accuracy.item(), dice_coefficient.item(), iou.item()

def test(weight_path, test_data_path, num_classes, result_save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data loader
    test_data_loader = DataLoader(MyDataset(test_data_path), batch_size=1, shuffle=False)

    # Create model instance
    "chosse net"
    model = CE_Trans(num_classes).to(device)
    # model = CE_TransUNet(num_classes).to(device)
    # model = UNet(num_classes).to(device)
    # model = Attention_unet(num_classes).to(device)
    # model = mit_PLD_b2(num_classes).to(device)
    # model = MTUNet(num_classes).to(device)
    # model = Build_LeViT_UNet_128s(num_classes, pretrained=False).to(device)

    # Load pre-trained weights
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # Create result save folder
    os.makedirs(result_save_path, exist_ok=True)

    # Initialize variables for performance metrics
    total_accuracy = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_samples = 0

    # Print the name of the current weight file
    print(f"Processing weight file: {os.path.basename(weight_path)}")

    # Redirect stdout to a log file
    log_file_path = os.path.join(result_save_path, 'log.txt')
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file  # Redirect stdout to the log file

        # Iterate through test data and evaluate the model
        for i, (image, segment_image) in enumerate(test_data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            # Use the model for inference
            with torch.no_grad():
                out_image = model(image)

            # Calculate metrics
            accuracy, dice_coefficient, iou = compute_metrics(out_image, segment_image.long())
            total_accuracy += accuracy
            total_dice += dice_coefficient
            total_iou += iou

            num_samples += 1

            print(f'Sample {i + 1}: Accuracy: {accuracy * 100:.2f}% Dice: {dice_coefficient:.4f} IoU: {iou:.4f}')

            # Save test result image
            result_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255
            result_image = result_image.byte()
            result_image_pil = Image.fromarray(result_image.squeeze().cpu().numpy(), 'L')

            # Extract file name from test data path
            test_image_name = test_data_loader.dataset.get_image_name(i)
            # Save result image with the same name as the test image
            result_image_pil.save(os.path.join(result_save_path, f'{test_image_name[:-4]}_result.png'))

        # Calculate average accuracy, Dice coefficient, and IoU
        average_accuracy = total_accuracy / num_samples
        average_dice = total_dice / num_samples
        average_iou = total_iou / num_samples

        print(f'Average Accuracy: {average_accuracy * 100:.2f}%')
        print(f'Average Dice: {average_dice:.4f}')
        print(f'Average IoU: {average_iou:.4f}')

        # Restore stdout
        sys.stdout = sys.__stdout__

    return average_accuracy, average_dice, average_iou

def extract_metrics_from_logs(result_base_path):
    epochs = []
    dices = []
    ious = []
    accuracies = []

    # List all result folders
    result_folders = [f for f in os.listdir(result_base_path) if os.path.isdir(os.path.join(result_base_path, f))]

    # Iterate through result folders
    for result_folder in result_folders:
        log_file_path = os.path.join(result_base_path, result_folder, 'log.txt')

        # Extract epoch number from the log file name
        epoch_match = re.search(r'epoch_(\d+)', result_folder)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        else:
            epoch = -1

        # Read the last three lines of the log file
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()
            last_three_lines = lines[-3:]

        # Extract metrics from the last three lines
        accuracy_match = re.search(r'Accuracy: (\d+\.\d+)%', last_three_lines[0])
        dice_match = re.search(r'Dice: (\d+\.\d+)', last_three_lines[1])
        iou_match = re.search(r'IoU: (\d+\.\d+)', last_three_lines[2])

        if accuracy_match and dice_match and iou_match:
            accuracy = float(accuracy_match.group(1)) / 100.0  # Convert to decimal
            dice = float(dice_match.group(1))
            iou = float(iou_match.group(1))

            epochs.append(epoch)
            accuracies.append(accuracy)
            dices.append(dice)
            ious.append(iou)

    return epochs, dices, ious, accuracies

def plot_metrics(epochs, dices, ious, accuracies, result_save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, dices, label='Dice')
    plt.plot(epochs, ious, label='IoU')
    plt.plot(epochs, accuracies, label='Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plot_image_path = os.path.join(result_save_path, 'metrics_plot.png')
    plt.savefig(plot_image_path)
    plt.close()

    print(f"Metrics plot saved at: {plot_image_path}")

if __name__ == '__main__':
    weight_folder = r'params'  # Replace with the folder containing your weight files
    test_data_path = r'C:\Users\Lenovo\Documents\GitHub\CE-TransUNet\Models\CE-TransUNet\test_data'  # Replace with your test data path
    num_classes = 2  # Replace with your number of classes
    result_base_path = 'test_results'  # Base folder path to save test result images

    # Create result base folder
    os.makedirs(result_base_path, exist_ok=True)

    # Run tests
    for weight_file in os.listdir(weight_folder):
        if weight_file.endswith('.pth'):
            weight_path = os.path.join(weight_folder, weight_file)
            result_save_path = os.path.join(result_base_path, weight_file.split('.')[0])  # Use weight file name for result folder

            average_accuracy, average_dice, average_iou = test(weight_path, test_data_path, num_classes, result_save_path)

    # Extract metrics from logs
    epochs, dices, ious, accuracies = extract_metrics_from_logs(result_base_path)

    # Plot metrics and save the plot
    plot_metrics(epochs, dices, ious, accuracies, result_base_path)
