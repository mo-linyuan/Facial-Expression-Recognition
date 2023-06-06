import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
now=datetime.now()
time=now.strftime("%d-%H-%M")

# 导入数据
def load_file(filename):
    # 打开文件
    data_file = open(filename, 'rb')
    # 读取数据
    data = pickle.load(data_file)
    # 关闭文件
    data_file.close()
    # 返回数据
    return data.history


# 绘制训练过程中的损失函数
def plot_loss(his, ds):
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(his['loss'])), his['loss'], label='train loss')
    plt.plot(np.arange(len(his['val_loss'])), his['val_loss'], label='valid loss')
    plt.legend(loc='best')
    plt.xlabel('epoch', fontsize=10.5)
    plt.ylabel('loss', fontsize=10.5)
    plt.tight_layout()
    plt.savefig('./assets/{}_loss_{}.png'.format(ds,time),dpi=600)


# 绘制训练过程中的准确率
def plot_acc(his, ds):
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(his['accuracy'])), his['accuracy'], label='trian accuracy')
    plt.plot(np.arange(len(his['val_accuracy'])), his['val_accuracy'], label='valid accuracy')
    plt.legend(loc='best')
    plt.xlabel('epoch', fontsize=10.5)
    plt.ylabel('accuracy', fontsize=10.5)
    plt.tight_layout()
    plt.savefig('./assets/{}_acc_{}.png'.format(ds,time),dpi=600)
    # 保存训练数据
    his_df = pd.DataFrame(his)
    his_df.to_csv('./train_history/{}_{}.csv'.format(ds,time))
