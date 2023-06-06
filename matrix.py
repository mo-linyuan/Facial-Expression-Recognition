# confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Contempt']
# classes = ['Angrer', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprised']
# confusion_matrix = np.array([(2356,329,72,0,6,286,6),
#                              (77,325,12,0,0,122,10),
#                              (414,18,2789,0,1,782,1129),
#                              (4,456,0,7790,478,30,4),
#                              (582,673,156,503,4089,100,89),
#                              (967,973,56,3,500,3589,89),
#                              (329,206,272,40,6,286,2856)],
#                             dtype=np.int)  # 输入特征矩阵
confusion_matrix = np.array([(125,1,0,0,6,0,0),
                             (3,172,0,0,0,0,0),
                             (0,1,74,0,0,0,0),
                             (0,0,0,207,0,0,0),
                             (1,0,0,0,80,3,0),
                             (0,0,0,3,0,246,0),
                             (1,0,0,0,2,0,51)],
                            dtype=np.int)  # 输入特征矩阵
proportion = []
for i in confusion_matrix:
    for j in i:
        temp = j / (np.sum(i))
        proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
# print(proportion)
pshow = []
for i in proportion:
    pt = "%.2f%%" % (i * 100)
    pshow.append(pt)
proportion = np.array(proportion).reshape(7, 7)  # reshape(列的长度，行的长度)
pshow = np.array(pshow).reshape(7, 7)
# print(pshow)
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=10.5, rotation=45)
plt.yticks(tick_marks, classes, fontsize=10.5)

thresh = confusion_matrix.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# ij配对，遍历矩阵迭代器
iters = np.reshape([[[i, j] for j in range(7)] for i in range(7)], (confusion_matrix.size, 2))
for i, j in iters:
    if (i == j):
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10.5, color='white',
                 weight=5)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10.5, color='white')
    else:
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10.5)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10.5)

plt.ylabel('True Expression', fontsize=10.5)
plt.xlabel('Predict Expression', fontsize=10.5)
plt.tight_layout()
plt.savefig('./assets/ck+_confusion_matrix.png', dpi=600)

