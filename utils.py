from tensorflow.keras.preprocessing.image import load_img, img_to_array


def get_fer2013_images():
    """
    从csv文件得到图片集
    :return:
    """
    import pandas as pd
    import numpy as np
    import scipy.misc as sm
    import os
    # 定义7种表情
    emotions = {
        '0': 'anger',  # 生气
        '1': 'disgust',  # 厌恶
        '2': 'fear',  # 恐惧
        '3': 'happy',  # 开心
        '4': 'sad',  # 伤心
        '5': 'surprised',  # 惊讶
        '6': 'neutral',  # 中性
    }

    def save_image_from_fer2013(file):
        faces_data = pd.read_csv(file)
        root = '../data/fer2013/'
        # 文件主要三个属性，emotion为表情列，pixels为像素数据列，usage为数据集所属列
        data_number = 0
        for index in range(len(faces_data)):
            # 解析每一行csv文件内容
            emotion_data = faces_data.loc[index][0]  # emotion
            image_data = faces_data.loc[index][1]  # pixels
            usage_data = faces_data.loc[index][2]  # usage
            # 将图片数据转换成48*48
            image_array = np.array(list(map(float, image_data.split()))).reshape((48, 48))

            folder = root + usage_data
            emotion_name = emotions[str(emotion_data)]
            image_path = os.path.join(folder, emotion_name)
            if not os.path.exists(folder):
                os.mkdir(folder)
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            # 图片文件名
            image_file = os.path.join(image_path, str(index) + '.jpg')
            sm.toimage(image_array).save(image_file)
            data_number += 1
        print('总共有' + str(data_number) + '张图片')

    save_image_from_fer2013('../data/fer2013/fer2013.csv')

def expression_analysis(distribution_possibility):
    """
    根据概率分布显示直方图
    :param distribution_possibility:
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    # 定义8种表情
    emotions = {
        '0': 'anger',
        '1': 'disgust',
        '2': 'fear',
        '3': 'happy',
        '4': 'sad',
        '5': 'surprised',
        '6': 'neutral',
        '7': 'contempt'
    }
    y_position = np.arange(len(emotions))
    plt.figure()
    plt.bar(y_position, distribution_possibility, align='center', alpha=0.5)
    plt.xticks(y_position, list(emotions.values()))
    plt.ylabel('possibility')
    plt.title('predict result')
    if not os.path.exists('../results'):
        os.mkdir('../results')
    plt.show()
    # plt.savefig('../results/rst.png')


def load_test_image(path):
    """
    读取外部测试图片
    :param path:
    :return:
    """
    img = load_img(path, target_size=(48, 48), color_mode="grayscale")
    img = img_to_array(img) / 255.
    return img


def index2emotion(index=0, kind='cn'):
    """
    根据表情下标返回表情字符串
    :param index:
    :return:
    """
    emotions = {
        '发怒': 'anger',
        '厌恶': 'disgust',
        '恐惧': 'fear',
        '开心': 'happy',
        '伤心': 'sad',
        '惊讶': 'surprised',
        '中性': 'neutral',
        '蔑视': 'contempt'

    }
    return list(emotions.values())[index]


def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=25):
    """
    :param img:
    :param text:
    :param left:
    :param top:
    :param text_color:
    :param text_size
    :return:
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype(
        "./assets/simsun.ttc", text_size, encoding="utf-8")  # 使用宋体
    draw.text((left, top), text, text_color, font=font_text)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_faces_from_gray_image(img_path):
    """
    获取图片中的人脸
    :param img_path:
    :return:
    """
    import cv2
    face_cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    # 遍历每一个脸
    faces_gray = []
    for (x, y, w, h) in faces:
        face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
        face_img_gray = cv2.resize(face_img_gray, (48, 48))
        faces_gray.append(face_img_gray)
    return faces_gray


def get_feature_map(model, layer_index, channels, input_img=None):
    """
    可视化每个卷积层学到的特征图
    :param model:
    :param layer_index:
    :param channels:
    :param input_img:
    :return:
    """
    if not input_img:
        input_img = load_test_image('../data/demo.jpg')
        input_img.shape = (1, 48, 48, 1)
    from keras import backend as K
    layer = K.function([model.layers[0].input], [model.layers[layer_index+1].output])
    feature_map = layer([input_img])[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 8))
    for i in range(channels):
        img = feature_map[:, :, :, i]
        plt.subplot(4, 8, i+1)
        plt.imshow(img[0], cmap='gray')
    plt.show()


if __name__ == '__main__':
    from model import CNN3
    model = CNN3()
    model.load_weights('../models/cnn3_best_weights.h5')
    get_feature_map(model, 1, 32)

