# MLP-Mixer方法的面部表情识别设计与实现

## 0.关于本项目

这个项目是我的毕业设计， 采用MLP-Mixer模型对面部表情识别进行实验，虽然结果并不算优秀，但不继续拘束于CNN，而是尝试一种新的模型范式，可能是本文的一点意义所在。本文还存在一些不足，未来还需做进一步探讨和改进。

## 1.MLP-Mixer

MLP-Mixer是谷歌在2021年提出了一种采用多层感知机替代CNN中的卷积操作和Transformer中的自注意力机制的网络结构。在大型数据集上训练时或者采用现代正则化方案时，MLP - Mixer在图像分类基准上获得了有竞争力的性能表现。

![image-20230606161610356](https://picgo-mly.obs.cn-north-4.myhuaweicloud.com/images/image-20230606161610356.png)

## 2.数据集准备

- CK+数据集
- Fer2013数据集

以上两个数据集均是面部表情识别项目常用的数据集，下载完成后，将它们放到新建的dataset文件夹中，dataset文件夹和本项目文件夹同一级别

## 3.模型训练

**MLP-Mixer网络结构的设置：**

model.py中设置网络结构，不同的网络结构有着不同的性能表现，程序中给出的是本个项目的默认结构。

**模型训练参数设置：**

在train.py中选择数据集、设置训练轮次和批次大小。

```
parser.add_argument("--dataset", type=str, default="fer2013")  # 选择数据集
parser.add_argument("--epochs", type=int, default=200)  # 训练轮数
parser.add_argument("--batch_size", type=int, default=32)  # 批次大小
```

设置学习率、优化器、编译参数、训练策略(早停法、学习率下降策略等)

```
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
```

## 4.训练结果

训练完成后从train.log可以看到模型在测试集上的识别准确率,在assets文件夹中生成了当前数据集的训练过程中的识别准确率和识别损失

![image-20230606161928970](https://picgo-mly.obs.cn-north-4.myhuaweicloud.com/images/image-20230606161928970.png)

## 5.系统设计

使用PyQT设计了一个面部表情识别系统，分别可以实现三种功能：

- 静态图像识别

![image-20230606162356943](https://picgo-mly.obs.cn-north-4.myhuaweicloud.com/images/image-20230606162356943.png)

- 摄像头识别

省略

- 识别识别

![image-20230606162612848](https://picgo-mly.obs.cn-north-4.myhuaweicloud.com/images/image-20230606162612848.png)