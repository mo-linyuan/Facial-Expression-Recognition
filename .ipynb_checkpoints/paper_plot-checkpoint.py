# ----绘制论文图像----#
import numpy as np
import matplotlib.pyplot as plt

with open('train_history/ck+_best.csv', 'r') as f:
    his = f.readlines()
    his = [i.strip().split(',') for i in his]
    his = np.array(his[1:], dtype=np.float32)
    print(his.shape)
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his[:, 1])), his[:, 1], label='train loss')
    plt.plot(np.arange(len(his[:, 2])), his[:, 2], label='valid loss')
    plt.legend(loc='best')
    plt.savefig('./paper_img/test_loss.png')





