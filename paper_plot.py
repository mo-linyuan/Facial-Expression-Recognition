# ----绘制论文图像----#
import numpy as np
import matplotlib.pyplot as plt

with open('train_history/fer2013_09-40.csv', 'r') as f:
    his = f.readlines()
    his = [i.strip().split(',') for i in his]
    his = np.array(his[1:], dtype=np.float32)
    print(his.shape)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(his[:, 1])), his[:, 1], label='train loss')
    plt.plot(np.arange(len(his[:, 3])), his[:, 3], label='valid loss')
    plt.legend(loc='best')
    plt.xlabel('训练轮次', fontsize=10.5, fontname='SimHei')
    plt.ylabel('准确率', fontsize=10.5, fontname='SimHei')
    plt.tight_layout()
    plt.savefig('./paper_img/test_accuracy.png',dpi=600)

with open('train_history/fer2013_8.csv', 'r') as f:
    his = f.readlines()
    his = [i.strip().split(',') for i in his]
    his = np.array(his[1:], dtype=np.float32)
    print(his.shape)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(his[:, 2])), his[:, 2], label='train loss')
    plt.plot(np.arange(len(his[:, 4])), his[:, 4], label='valid loss')
    plt.legend(loc='best')
    plt.xlabel('训练轮次', fontsize=10.5, fontname='SimHei')
    plt.ylabel('损失值', fontsize=10.5, fontname='SimHei')
    plt.tight_layout()
    plt.savefig('./paper_img/test_loss.png',dpi=600)







