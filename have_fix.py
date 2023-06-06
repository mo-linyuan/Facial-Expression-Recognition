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
from data import Fer2013, Jaffe, CK
from sklearn.model_selection import train_test_split
from model import MLPMixer
from plot import plot_loss, plot_acc

# 使用GPU：
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 读取命令行参数
parser = argparse.ArgumentParser()
# 选择数据集
parser.add_argument("--dataset", type=str, default="ck+")
# 训练轮数
parser.add_argument("--epochs", type=int, default=200)
# 批次大小
parser.add_argument("--batch_size", type=int, default=32)
# 是否绘制训练过程中的损失和准确率
parser.add_argument("--plot_history", type=bool, default=True)
# 读取命令行参数,用于后续打印
opt = parser.parse_args()
his = None
print(opt)
# Mixer Layer的深度
num_blocks = 8.1

# 输出日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()
file_hander = logging.FileHandler('train.log')
file_hander.setLevel(logging.INFO)
logger.addHandler(file_hander)

# 设置早停法,如果验证集的损失在50个epoch内没有下降，则停止训练
early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, verbose=1)

# 如果选择的数据集jaffe
if opt.dataset == 'jaffe':
    # 读取数据集
    expressions, x, y = Jaffe().generate_train()
    # target编码，将表情名称转换为数字
    # to_categorical: 将类别向量转换为二进制（只有0和1）的矩阵类型表示
    # reshape: 将数据转换为指定的维度
    # reshape(y_train.shape[0], -1):中的-1表示自动计算列数
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y = np.hstack((y, np.zeros((y.shape[0], 1))))

    # 划分训练集验证集, 80%训练集，20%验证集，rnandom_state=2019，保证每次划分的结果一致，方便调参；如果不设置，每次划分的结果都不一样
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)
    logger.info('load jaffe dataset successfully, train: {}, valid: {}'.format(y_train.shape[0], y_valid.shape[0]))

    # 数据实时增强
    train_generator = ImageDataGenerator(rotation_range=5,  # 以度为单位的随机旋转
                                         width_shift_range=0.01,  # 水平移动
                                         height_shift_range=0.01,  # 垂直移动
                                         horizontal_flip=True,  # 水平翻转
                                         shear_range=0.1,  # 变换角度
                                         zoom_range=0.1).flow(x_train, y_train, batch_size=opt.batch_size)  # 随机缩放
    # 验证集不需要增强
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

    model = MLPMixer(input_shape=(48, 48, 1),  # 图像的大小和通道数
                     num_classes=8,  # 分类数
                     num_blocks=8,  # Mixer Layer数量
                     patch_size=8,  # 图像块大小，8x8
                     hidden_dim=256,  # 图像块MLP和通道MLP的输入和输出维度
                     tokens_mlp_dim=2048,
                     channels_mlp_dim=256,
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
        ModelCheckpoint('./models/mlp_weights.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True), early_stopping]

    # 训练模型
    history_jaffe = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                              validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                              callbacks=callback)
    his = history_jaffe
    logger.info('Model trained successful.')
    #在验证集上评估模型
    metrics = model.evaluate(valid_generator)
    logger.info(f'num_blocks={num_blocks} -- Loss:{metrics[0]}, Accuracy:{metrics[1]}')


elif opt.dataset == 'ck+':
    expr, x, y = CK().generate_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 划分训练集验证集
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)
    logger.info("load CK+ dataset successfully, train: {}, valid: {}".format(y_train.shape[0],
                                                                             y_valid.shape[0]))
    train_generator = ImageDataGenerator(rotation_range=10,
                                         width_shift_range=0.05,
                                         height_shift_range=0.05,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         zoom_range=0.2).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

    model = MLPMixer(input_shape=(48, 48, 1),  # 图像的大小和通道数
                     num_classes=8,  # 分类数，本来是7，但是源代码写8，这里暂时不做深究
                     num_blocks=8,  # MLP-Mixer层的深度
                     patch_size=8,  # MLP-Mixer层中每个位置对应的图像区域的大小
                     hidden_dim=512,  # 图像块MLP和通道MLP的输入和输出维度
                     tokens_mlp_dim=2048,
                     channels_mlp_dim=256,
                     use_softmax=True)
    adam = Adam(lr=0.0001)
    # 编译模型，指定损失函数和优化器
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info('Model compiled successful.')
    # 保存最好的模型
    callback = [
        ModelCheckpoint('./models/{}.h5'.format(opt.dataset), monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True)]
    # 训练模型
    history_ck = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                     validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                     callbacks=callback)
    his = history_ck
    logger.info('Model trained successful.')
    # 评估模型
    metrics = model.evaluate(valid_generator)
    logger.info(f'num_blocks={num_blocks} -- Loss:{metrics[0]}, Accuracy:{metrics[1]}')

# 如果选择的数据集是fer2013
elif opt.dataset == "fer2013":
    # 使用generate_train()方法，返回的是表情名称，训练集，训练集标签
    expressions, x_train, y_train = Fer2013().generate_train()
    # 使用generate_valid()方法，返回的是表情名称，验证集，验证集标签
    _, x_valid, y_valid = Fer2013().generate_valid()
    # 使用generate_test()方法，返回的是表情名称，测试集，测试集标签
    _, x_test, y_test = Fer2013().generate_test()

    y_train = to_categorical(y_train).reshape(y_train.shape[0], -1)
    y_valid = to_categorical(y_valid).reshape(y_valid.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y_train = np.hstack((y_train, np.zeros((y_train.shape[0], 1))))
    y_valid = np.hstack((y_valid, np.zeros((y_valid.shape[0], 1))))

    logger.info('load fer2013 dataset successfully, train: {}, valid: {}'.format(x_train.shape[0], x_valid.shape[0]))
    model = MLPMixer(input_shape=(48, 48, 1),  # 图像的大小和通道数
                     num_classes=8,  # 分类数，本来是7，但是源代码写8，这里暂时不做深究
                     num_blocks=8,  # Mixer Layer数量
                     patch_size=8,  # MLP-Mixer层中每个位置对应的图像区域的大小
                     hidden_dim=256,  # 图像块MLP和通道MLP的输入和输出维度
                     tokens_mlp_dim=256,
                     channels_mlp_dim=256,
                     use_softmax=True)
    # Adam是一种自适应学习率的优化算法，可以自动调整学习率，使得训练更快收敛
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info('Model compiled successful.')


    callback = [
        ModelCheckpoint('./models/{}.h5'.format(opt.dataset), monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True), early_stopping]

    train_generator = ImageDataGenerator(rotation_range=10,
                                         width_shift_range=0.05,
                                         height_shift_range=0.05,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         zoom_range=0.2).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)
    history_fer2013 = model.fit(train_generator,
                                steps_per_epoch=len(y_train) // opt.batch_size,
                                epochs=opt.epochs,
                                validation_data=valid_generator,
                                validation_steps=len(y_valid) // opt.batch_size,
                                callbacks=callback)
    his = history_fer2013
    logger.info('Model trained successful.')
    metrics = model.evaluate(valid_generator)
    logger.info(f'num_blocks={num_blocks} -- Loss:{metrics[0]}, Accuracy:{metrics[1]}')

# 绘制训练过程中的损失和准确率

if opt.plot_history:
    plot_loss(his.history, opt.dataset, num_blocks)
    plot_acc(his.history, opt.dataset, num_blocks)
