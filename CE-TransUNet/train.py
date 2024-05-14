import os
import tqdm
import time  # 添加time模块
import matplotlib.pyplot as plt
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import MyDataset
from torchvision.utils import save_image
from ce_net import *
from atUnet import *
from new_unet import *
from LeUNet import *
from ssformer import *
from MTUNet import *
from new_unet import *

def compute_accuracy(predictions, labels):
    with torch.no_grad():
        predicted_labels = torch.argmax(predictions, dim=1)
        correct_predictions = (predicted_labels == labels).float()
        accuracy = correct_predictions.mean()
    return accuracy.item()

def compute_dice_coefficient(predictions, labels):
    with torch.no_grad():
        predicted_labels = torch.argmax(predictions, dim=1)
        intersection = torch.sum(predicted_labels * labels)
        union = torch.sum(predicted_labels) + torch.sum(labels)
        dice_coefficient = (2.0 * intersection) / (union + 1e-7)
    return dice_coefficient.item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

<<<<<<< HEAD
def train(num_epochs, batch_size, num_classes, data_path, pretrained_weight_path, trained_weight_path, save_path, plot_interval=50, save_interval=10, learning_rate=0.001):
=======
def train(num_epochs, batch_size, num_classes, data_path, weight_path, save_path, plot_interval=1):
>>>>>>> 17ba0eda3f168109cd914db7b74b778e56411e54
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(MyDataset(data_path), batch_size=batch_size, shuffle=True)
    "chosse net"
    net = CE_VT(num_classes).to(device)
    # net = CE_TransUNet(num_classes).to(device)
    # net = UNet(num_classes).to(device)
    # net = Attention_unet(num_classes).to(device)
    # net = mit_PLD_b2(num_classes).to(device)
    # net = MTUNet(num_classes).to(device)
    # net = Build_LeViT_UNet_128s(num_classes, pretrained=False).to(device)
    # net = Attention_unet(num_classes).to(device)
    "choose net"




    if os.path.exists(pretrained_weight_path):
        net.load_state_dict(torch.load(pretrained_weight_path))
        print(f'Successfully loaded pre-trained weights from {pretrained_weight_path}!')
    else:
        print('Pre-trained weight file not found. Starting from scratch.')

    opt = optim.Adam(net.parameters(), lr=learning_rate)  # 设置学习率
    loss_fun = nn.CrossEntropyLoss()

    num_params = count_parameters(net)
    print(f'Number of trainable parameters in the model: {num_params}')

    # Clear the training logs file at the beginning of training
    with open('train_logs.txt', 'w') as f:
        f.write('')

    loss_values = []
    dice_values = []
    epoch_accuracies = []
    saved_weight_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()  # 记录开始时间

        total_loss = 0.0
        total_accuracy = 0.0
        total_dice = 0.0

        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image.long())

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            accuracy = compute_accuracy(out_image, segment_image.long())
            dice_coefficient = compute_dice_coefficient(out_image, segment_image)

            total_accuracy += accuracy
            total_dice += dice_coefficient
            total_loss += train_loss.item()

            if i % 1 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] Batch [{i + 1}/{len(data_loader)}] Loss: {train_loss.item()} Accuracy: {accuracy * 100:.2f}% Dice: {dice_coefficient:.4f}')

            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)

        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time  # 计算epoch训练时间

        average_loss = total_loss / len(data_loader)
        average_accuracy = total_accuracy / len(data_loader)
        average_dice = total_dice / len(data_loader)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {average_loss:.4f} Average Accuracy: {average_accuracy * 100:.2f}% Average Dice: {average_dice:.4f} Time: {epoch_time:.2f} seconds')

        loss_values.append(average_loss)
        dice_values.append(average_dice)

        average_epoch_accuracy = total_accuracy / len(data_loader)
        epoch_accuracies.append(average_epoch_accuracy)

        if (epoch + 1) % plot_interval == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), loss_values, label='Loss', marker='o')
            plt.plot(range(1, epoch + 2), dice_values, label='Dice', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)

            plot_save_dir = 'loss_dice_plots'
            os.makedirs(plot_save_dir, exist_ok=True)

            plot_save_path = os.path.join(plot_save_dir, f'loss_dice_plot_epoch_{epoch + 1}.png')
            plt.savefig(plot_save_path)
            plt.show()

        if (epoch + 1) % save_interval == 0:
            saved_weight_counter += 1
            saved_weight_path = os.path.join(trained_weight_path, f'epoch_{saved_weight_counter}.pth')
            torch.save(net.state_dict(), saved_weight_path)
            print(f'Saved checkpoint at epoch {epoch + 1} successfully to {saved_weight_path}!')

        with open('train_logs.txt', 'a') as f:
            f.write(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {average_loss:.4f} Average Accuracy: {average_accuracy * 100:.2f}% Average Dice: {average_dice:.4f} Time: {epoch_time:.2f} seconds\n')

    final_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
    print(f'Final Accuracy: {final_accuracy * 100:.2f}%')


if __name__ == '__main__':
<<<<<<< HEAD
    num_epochs = 2
=======
    num_epochs = 1
>>>>>>> 17ba0eda3f168109cd914db7b74b778e56411e54
    batch_size = 1
    num_classes = 2
    data_path = r'data'
    pretrained_weight_path = 'pretrained_params/ce_transunet.pth'
    trained_weight_path = 'params'
    save_path = 'train_image'

    # 在这里设置你想要的学习率
    learning_rate = 0.01
    train(num_epochs, batch_size, num_classes, data_path, pretrained_weight_path, trained_weight_path, save_path, learning_rate=learning_rate)
