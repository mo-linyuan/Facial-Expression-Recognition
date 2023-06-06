# 模型训练
import os
import argparse
import logging
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from data import Fer2013, CK
from sklearn.model_selection import train_test_split
from model import MLPMixer
from plot import plot_loss, plot_acc
from datetime import datetime

now = datetime.now()
time = now.strftime("%d-%H-%M")

# 使用GPU：
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()  # 读取命令行参数
parser.add_argument("--dataset", type=str, default="fer2013")  # 选择数据集
parser.add_argument("--epochs", type=int, default=200)  # 训练轮数
parser.add_argument("--batch_size", type=int, default=32)  # 批次大小
parser.add_argument("--plot_history", type=bool, default=True)  # 是否绘制训练过程中的损失和准确率

opt = parser.parse_args()  # 读取命令行参数,用于后续打印
his = None
print(opt)

# 输出日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()
file_hander = logging.FileHandler('train.log')
file_hander.setLevel(logging.INFO)
logger.addHandler(file_hander)

if opt.dataset == 'fer2013':
    expressions, x_train, y_train = Fer2013().generate_train()
    _, x_valid, y_valid = Fer2013().generate_valid()
    _, x_test, y_test = Fer2013().generate_test()
    # target编码
    y_train = to_categorical(y_train).reshape(y_train.shape[0], -1)
    y_valid = to_categorical(y_valid).reshape(y_valid.shape[0], -1)
    y_test = to_categorical(y_test).reshape(y_test.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y_train = np.hstack((y_train, np.zeros((y_train.shape[0], 1))))
    y_valid = np.hstack((y_valid, np.zeros((y_valid.shape[0], 1))))
    y_test = np.hstack((y_test, np.zeros((y_test.shape[0], 1))))

# Fer2013数据集的Training数据集
elif opt.dataset == 'fer2013_Training':
    expressions, x_train, y_train = Fer2013().generate_train()
    # target编码
    y_train = to_categorical(y_train).reshape(y_train.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y_train = np.hstack((y_train, np.zeros((y_train.shape[0], 1))))
    # 划分训练集验证集, 60%训练集，20%验证集，20%测试集
    x_train, x_temp, y_train, y_temp = train_test_split(x_train, y_train, test_size=0.4, random_state=2023)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=2023)

# Fer2013数据集的PublicTest数据集
elif opt.dataset == 'fer2013_PublicTest':
    expressions, x, y = Fer2013().generate_valid()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y = np.hstack((y, np.zeros((y.shape[0], 1))))
    # 划分训练集验证集, 60%训练集，20%验证集，20%测试集
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=2023)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=2023)

# Fer2013数据集的PrivateTest数据集
elif opt.dataset == 'fer2013_PrivateTest':
    expressions, x, y = Fer2013().generate_test()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y = np.hstack((y, np.zeros((y.shape[0], 1))))
    # 划分训练集验证集, 60%训练集，20%验证集，20%测试集
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=2023)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=2023)


elif opt.dataset == 'ck+':
    expressions, x, y = CK().generate_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 划分训练集验证集, 60%训练集，20%验证集，20%测试集
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=2023)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=2023)

logger.info(
    f'load {opt.dataset} dataset successfully, train: {y_train.shape[0]}, valid: {y_valid.shape[0]}, test: {y_test.shape[0]}')
model = MLPMixer()
# model.summary()
adam = Adam(lr=0.001)
# 编译模型
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
logger.info('Model compiled successful.')
# 设置早停法,如果验证集的损失在20个epoch内没有下降，则停止训练
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1)
# # 设置学习率下降策略,如果验证集的损失在20个epoch内没有下降，则学习率降低为原来的0.1倍
# reducelronplateau=ReduceLROnPlateau(monitor='lr', factor=0.1, patience=20, verbose=1)
callback = [
    ModelCheckpoint(f'./models/{opt.dataset}_weights_{time}.h5', monitor='val_accuracy', verbose=True,
                    save_best_only=True,
                    save_weights_only=True),
    early_stopping]
# 训练数据实时增强
train_generator = ImageDataGenerator(rotation_range=10,
                                     width_shift_range=0.05,
                                     height_shift_range=0.05,
                                     horizontal_flip=True,
                                     shear_range=0.2,
                                     zoom_range=0.2).flow(x_train, y_train, batch_size=opt.batch_size)

valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)
test_generator = ImageDataGenerator().flow(x_test, y_test, batch_size=opt.batch_size)
# 训练模型
history = model.fit(train_generator,
                    steps_per_epoch=len(y_train) // opt.batch_size,
                    epochs=opt.epochs,
                    validation_data=valid_generator,
                    validation_steps=len(y_valid) // opt.batch_size,
                    callbacks=callback)
his = history
logger.info('Model trained successful.')
# 在测试集上评估模型
metrics = model.evaluate(test_generator)
logger.info(f'{time} - {opt.dataset}--Loss:{metrics[0]}, Accuracy:{metrics[1]}')

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print("test accuacy", np.sum(pred.reshape(-1) == y_test.reshape(-1)) / y_test.shape[0])

# 绘制训练过程中的损失和准确率
if opt.plot_history:
    plot_loss(his.history, opt.dataset)
    plot_acc(his.history, opt.dataset)
