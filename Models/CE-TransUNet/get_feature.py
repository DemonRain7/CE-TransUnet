import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

class getfeature(nn.Module):

    def __init__(self, dim, qkv_bias=True):
        super(getfeature, self).__init__()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, mask=None):
        x = x.permute(1, 0, 2, 3)
        x = x.reshape(x.size(0), -1)
        x = x.permute(1, 0)
        qkv = self.qkv(x)
        return qkv

if __name__ == '__main__':
    'test'

    image_path = r"C:\Users\Lenovo\Desktop\IMG_20240306_152249.jpg"
    image = keep_image_size_open_rgb(image_path)
    x = transform(image)
    x = x.reshape(1, 3, 224, 224)
    model = getfeature(3)
    output = model(x)
    output = output.reshape(3, 3, 1, 224, 224)
    q, k, v = output.unbind(0)

    # 计算注意力
    q = q / 16
    attn = (q @ k.transpose(-2, -1))
    print(attn)
    Soft = nn.Softmax(dim=-1)
    attn = Soft(attn)
    r, g, b = attn.unbind(0)
    print("r", r)
    print("g", g)
    print("b", b)

    # 定义划分的区间
    bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 1]

    # 计算每个量级中的数据个数
    hist_r, _ = np.histogram(r.cpu().detach().numpy().flatten(), bins=bins)
    hist_g, _ = np.histogram(g.cpu().detach().numpy().flatten(), bins=bins)
    hist_b, _ = np.histogram(b.cpu().detach().numpy().flatten(), bins=bins)

    # 打印结果
    print("Histogram for attn_r:")
    for i in range(len(bins) - 1):
        print(f"({bins[i]}, {bins[i + 1]}]: {hist_r[i]}")

    print("\nHistogram for attn_g:")
    for i in range(len(bins) - 1):
        print(f"({bins[i]}, {bins[i + 1]}]: {hist_g[i]}")

    print("\nHistogram for attn_b:")
    for i in range(len(bins) - 1):
        print(f"({bins[i]}, {bins[i + 1]}]: {hist_b[i]}")
