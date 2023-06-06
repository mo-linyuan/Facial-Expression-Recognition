"""
date: 2023-04-15
describe: 用于数据的读取和处理
"""
# tqdm是一个用于迭代过程中显示进度条的工具库
from tqdm import tqdm
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split


# fer2013数据集
class Fer2013(object):
    def __init__(self, folder="../dataset/fer"):
        self.folder = folder

    def generate_train(self):
        """
        产生训练数据
        :return experssions: 读取文件的顺序与标签的下标对应
        :return x_train: 训练数据集
        :return y_train: 训练标签
        """
        # 设置训练目录
        folder = os.path.join(self.folder, 'Training')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        # 解释下列代码
        # 读取训练集

        for i in tqdm(range(len(expressions))):
            # contempt不是fer2013数据集的表情之一，作为结束标志
            if expressions[i] == 'contempt':
                continue
            expressions_folder = os.path.join(folder, expressions[i])
            # 读取每个表情文件夹下的图片
            images = os.listdir(expressions_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expressions_folder, images[j]), target_size=(48, 48),
                               color_mode="grayscale")
                # 灰度化
                img = img_to_array(img)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def generate_valid(self):
        """
        产生验证集数据
        :return experssions: 读取文件的顺序与标签的下标对应
        :return x_valid: 验证数据集
        :return y_valid: 验证标签
        """
        # 设置训练目录
        folder = os.path.join(self.folder, 'PublicTest')
        # contempt不是表情之一，作为后续遍历表情的结束标志
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_valid = []
        y_valid = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_valid.append(img)
                y_valid.append(i)
        x_valid = np.array(x_valid).astype('float32') / 255.
        y_valid = np.array(y_valid).astype('int')
        return expressions, x_valid, y_valid

    def generate_test(self):
        """
        产生测试集数据
        :return experssions: 读取文件的顺序与标签的下标对应
        :return x_valid: 测试数据集
        :return y_valid: 测试标签
        """
        folder = os.path.join(self.folder, 'PrivateTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_test = []
        y_test = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_test.append(img)
                y_test.append(i)
        x_test = np.array(x_test).astype('float32') / 255.
        y_test = np.array(y_test).astype('int')
        return expressions, x_test, y_test

class CK(object):
    """
    CK+没有测试数据，需要自己划分
    """

    def __init__(self):
        self.folder = '../dataset/ck+'

    def generate_train(self):
        """
        产生训练数据
        :return:
        """
        folder = self.folder
        # 为了模型训练统一，这里加入neural
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'neutral':
                # 没有中性表情，直接跳过
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train
