# 模型训练
import os
import argparse
import logging
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from data import Fer2013, CK
from sklearn.model_selection import train_test_split
from model import MLPMixer
from plot import plot_loss, plot_acc
from datetime import datetime
now=datetime.now()
time=now.strftime("%H-%M-%S")

# 使用GPU：
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()  # 读取命令行参数
parser.add_argument("--dataset", type=str, default="ck+")  # 选择数据集
parser.add_argument("--epochs", type=int, default=200)  # 训练轮数
parser.add_argument("--batch_size", type=int, default=64)  # 批次大小
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

# 设置早停法,如果验证集的损失在50个epoch内没有下降，则停止训练
early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, verbose=1)

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

elif opt.dataset == 'ck+':
    expressions, x, y = CK().generate_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 划分训练集验证集, 60%训练集，20%验证集，20%测试集
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=2023)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=2023)

logger.info(f'load {opt.dataset} dataset successfully, train: {y_train.shape[0]}, valid: {y_valid.shape[0]}, test: {y_test.shape[0]}')
# 训练数据实时增强
train_generator = ImageDataGenerator(rotation_range=5,  # 以度为单位的随机旋转
                                     width_shift_range=0.01,  # 水平移动
                                     height_shift_range=0.01,  # 垂直移动
                                     horizontal_flip=True,  # 水平翻转
                                     shear_range=0.1,  # 变换角度
                                     zoom_range=0.1).flow(x_train, y_train, batch_size=opt.batch_size)  # 随机缩放

valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)
test_generator = ImageDataGenerator().flow(x_test, y_test, batch_size=opt.batch_size)

model = MLPMixer(input_shape=(48, 48, 1),  # 图像的大小和通道数
                 num_classes=8,  # 分类数
                 num_blocks=8,  # Mixer Layer数量
                 patch_size=8,  # 图像块大小，8x8
                 hidden_dim=512,  # 图像块MLP和通道MLP的输入和输出维度
                 tokens_mlp_dim=256,
                 channels_mlp_dim=2048,
                 use_softmax=True)
# model.summary()
adam = Adam(lr=0.001)
# 编译模型
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
logger.info('Model compiled successful.')

# 详细解释下面的callback以及其中的参数
# ModelCheckpoint: 保存模型的回调函数，该回调函数在每个epoch后保存模型到filepath
# filepath: 字符串，保存模型的路径
# monitor: 被监测的数据
# verbose: 详细信息模式，0或者1
# save_best_only: 如果save_best_only=True，当监测值有改进时才会保存当前的模型
callback = [
    ModelCheckpoint(f'./models/{opt.dataset}_weights.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                    save_weights_only=True), early_stopping]

# 训练模型
history = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                          validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                          callbacks=callback)
his = history
logger.info('Model trained successful.')
# 在测试集上评估模型
metrics = model.evaluate(test_generator)
logger.info(f'{time} - {opt.dataset}--Loss:{metrics[0]}, Accuracy:{metrics[1]}')

# 绘制训练过程中的损失和准确率
if opt.plot_history:
    plot_loss(his.history, opt.dataset)
    plot_acc(his.history, opt.dataset)
