import sys
import os
from PyQt5 import QtWidgets
from model import MLPMixer
from ui import UI
sys.path.append(os.path.dirname(__file__) + '/ui')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def CNN3(input_shape=(48, 48, 1), n_classes=8):
    """
    参考论文实现
    A Compact Deep Learning Model for Robust Facial Expression Recognition
    :param input_shape:
    :param n_classes:
    :return:
    """
    # input
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, \
        AveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import PReLU
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    # block1
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

def load_model():

    model = MLPMixer(input_shape=(48, 48, 1),  # 图像的大小和通道数
                     num_classes=8,  # 分类数
                     num_blocks=8,  # Mixer Layer数量
                     patch_size=8,  # 图像块大小，8x8
                     hidden_dim=512,  # 图像块MLP和通道MLP的输入和输出维度
                     tokens_mlp_dim=256,
                     channels_mlp_dim=2048,
                     use_softmax=True)
    # model = CNN3()
    # model.load_weights('./models/cnn3_best_weights.h5')
    model.load_weights('./models/ck+_weights.h5')
    return model


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
