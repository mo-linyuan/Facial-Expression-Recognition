from data import CK
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

expressions, x, y = CK().generate_train()
# target编码
y = to_categorical(y).reshape(y.shape[0], -1)
# 为了统一几个数据集，必须增加一列为0的
y = np.hstack((y, np.zeros((y.shape[0], 1))))
# 划分训练集和测试集
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)

print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)