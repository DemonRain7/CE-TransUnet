import torch
import torch.nn.functional as f
import torch.nn as nn

class Cat_block(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
    def forward(self, att, x):
        output = torch.cat((att, x), 1)
        output = self.conv(output)
        return output

class Conv_block(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
    def forward(self, input):
            x = self.conv(input)
            return x

class Up_block(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> None:
         super().__init__()
         self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
    def forward(self, input):
        x = self.up(input)

        return x

class Attention_skip(nn.Module):
    def __init__(self, x_channel, g_channel, out_channel) -> None:
         super().__init__()
         self.Wx = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(in_channels=x_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            
         )
         self.Wg = nn.Sequential(
            nn.Conv2d(in_channels=g_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
         )
         self.bias = nn.Sequential(
            nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
         )
    def forward(self, x, g):
        x1 = self.Wx(x)
        g1 = self.Wg(g)
        #print(x1.shape)
        #print(g1.shape)
        bias = x1 + g1
        bias = nn.ReLU(True)(bias)
        bias = nn.Upsample(scale_factor=2)(bias)
        bias = self.bias(bias)
        x = x * bias
        return x * bias

class Down_block(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, input):
        x = self.down(input)

        return x

class Attention_unet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.Conv1 = Conv_block(3, 64)
        self.Down1 = Down_block(64, 64)
        self.Conv2 = Conv_block(64, 128)
        self.Down2 = Down_block(128, 128)
        self.Conv3 = Conv_block(128, 256)
        self.Down3 = Down_block(256, 256)
        self.Conv4 = Conv_block(256, 512)
        self.Down4 = Down_block(512, 512)

        self.Conv5 = Conv_block(512, 1024)

        self.Up4 = Up_block(1024, 512)
        self.Conv4f = Conv_block(1024, 512)
        self.Up3 = Up_block(512, 256)
        self.Conv3f = Conv_block(512, 256)
        self.Up2 = Up_block(256, 128)
        self.Conv2f = Conv_block(256, 128)
        self.Up1 = Up_block(128, 64)
        self.Conv1f = Conv_block(128, 64)

        self.skip1 = Attention_skip(64, 128, 64)
        self.skip2 = Attention_skip(128, 256, 128)
        self.skip3 = Attention_skip(256, 512, 256)
        self.skip4 = Attention_skip(512, 1024, 512)

        self.out = nn.Conv2d(64, num_classes, 1, 1, 0)

    def forward(self, input):
        #print(input.shape)
        x = input
        x = self.Conv1(x)
        x1 = x

        x = self.Down1(x)
        x = self.Conv2(x)
        x2 = x

        x = self.Down2(x)
        x = self.Conv3(x)
        x3 = x

        x = self.Down3(x)
        x = self.Conv4(x)
        x4 = x

        x = self.Down4(x)
        x = self.Conv5(x)

        att4 = self.skip4(x4, x)
        x = self.Up4(x)
        x = torch.cat((att4, x), 1)
        x = self.Conv4f(x)
        
        att3 = self.skip3(x3, x)
        x = self.Up3(x)
        x = torch.cat((att3, x), 1)
        x = self.Conv3f(x)

        att2 = self.skip2(x2, x)
        x = self.Up2(x)
        x = torch.cat((att2, x), 1)
        x = self.Conv2f(x)
        
        att1 = self.skip1(x1, x)
        x = self.Up1(x)
        x = torch.cat((att1, x), 1)
        x = self.Conv1f(x)
        x = self.out(x)

        return  x

class demo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Conv1 = Conv_block(3, 64)
        self.Down1 = Down_block(64, 64)
        self.Conv2 = Conv_block(64, 128)
        self.Down2 = Down_block(128, 128)
        self.Conv3 = Conv_block(128, 256)
        self.Down3 = Down_block(256, 256)
        self.Conv4 = Conv_block(256, 512)
        self.Down4 = Down_block(512, 512)

        self.Conv5 = Conv_block(512, 1024)

        self.Up4 = Up_block(1024, 512)
        self.Conv4f = Conv_block(1024, 512)
        self.Up3 = Up_block(512, 256)
        self.Conv3f = Conv_block(512, 256)
        self.Up2 = Up_block(256, 128)
        self.Conv2f = Conv_block(256, 128)
        self.Up1 = Up_block(128, 64)
        self.Conv1f = Conv_block(128, 64)

        self.skip1 = Attention_skip(64, 128, 64)
        self.skip2 = Attention_skip(128, 256, 128)
        self.skip3 = Attention_skip(256, 512, 256)
        self.skip4 = Attention_skip(512, 1024, 512)

        self.out = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, input):
        #print(input.shape)
        x = input
        x = self.Conv1(x)
        x1 = x

        x = self.Down1(x)
        x = self.Conv2(x)
        x2 = x

        x = self.Down2(x)
        x = self.Conv3(x)
        x3 = x

        x = self.Down3(x)
        x = self.Conv4(x)
        
        att3 = self.skip3(x3, x)
        x = self.Up3(x)
        x = torch.cat((att3, x), 1)
        x = self.Conv3f(x)

        att2 = self.skip2(x2, x)
        x = self.Up2(x)
        x = torch.cat((att2, x), 1)
        x = self.Conv2f(x)
        
        att1 = self.skip1(x1, x)
        x = self.Up1(x)
        x = torch.cat((att1, x), 1)
        x = self.Conv1f(x)
        x = self.out(x)
        
        x = torch.sigmoid(x)
        
        
        return  x



class TMP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Conv1 = Conv_block(3, 64)
        self.Down1 = Down_block(64, 64)
        self.Conv2 = Conv_block(64, 128)
        self.Down2 = Down_block(128, 128)
        self.Conv3 = Conv_block(128, 256)
        self.Down3 = Down_block(256, 256)
        self.Conv4 = Conv_block(256, 512)
        self.Down4 = Down_block(512, 512)

        self.Conv5 = Conv_block(512, 1024)

        self.Up4 = Up_block(1024, 512)
        self.Conv4f = Conv_block(1024, 512)
        self.Up3 = Up_block(512, 256)
        self.Conv3f = Conv_block(512, 256)
        self.Up2 = Up_block(256, 128)
        self.Conv2f = Conv_block(256, 128)
        self.Up1 = Up_block(128, 64)
        self.Conv1f = Conv_block(128, 64)

        self.skip1 = Attention_skip(64, 128, 64)
        self.skip2 = Attention_skip(128, 256, 128)
        self.skip3 = Attention_skip(256, 512, 256)
        self.skip4 = Attention_skip(512, 1024, 512)

        self.out = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, input):
        #print(input.shape)
        x = input
        x = self.Conv1(x)
        x1 = x

        x = self.Down1(x)
        x = self.Conv2(x)
        x2 = x

        x = self.Down2(x)
        x = self.Conv3(x)


        att2 = self.skip2(x2, x)
        x = self.Up2(x)
        x = torch.cat((att2, x), 1)
        x = self.Conv2f(x)
        
        att1 = self.skip1(x1, x)
        x = self.Up1(x)
        x = torch.cat((att1, x), 1)
        x = self.Conv1f(x)
        x = self.out(x)

        x = torch.sigmoid(x)
        
        
        return  x
        
        