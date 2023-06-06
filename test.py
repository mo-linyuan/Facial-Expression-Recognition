import ui
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#数据准备
data = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]] #换成自己的2d数组
data = np.array(data)                             #list转numpy
data = torch.from_numpy(data)                     #numpy转tensor张量
data = torch.tensor(data, dtype=torch.float32)    # Height x Width 尺寸的数据
data = data.unsqueeze(0)                          # 升维度， 1 x Height x Width
data = data.unsqueeze(0)                          # 1 x 1 x Height x Width

#配置卷积参数
in_channels = 1
out_channels = 8
kernel_size = 3
Produce = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size, bias=False, stride=1, padding=kernel_size//2)

#卷积
img = Produce(data)                #得到  1  x  1  x  Height  x  Weight尺寸的矩阵(张量)

img = img.squeeze(0)[0]            #得到 Height x Weight  尺寸矩阵(张量)

img = img.detach().numpy()         #张量转Numpy

plt.imshow(img, origin='lower')
plt.show()