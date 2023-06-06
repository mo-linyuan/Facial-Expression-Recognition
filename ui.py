import time

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.use("Qt5Agg")
import sys

sys.path.append('../')
from recognition import *

font = QtGui.QFont()
font.setFamily("微软雅黑")

class UI(object):
    def __init__(self, form, model):
        self.setup_ui(form)
        self.model = model

    def setup_ui(self, form):
        form.setObjectName("Form")
        form.resize(1200, 900)
        # 原图无图时显示的label
        self.picture = QtWidgets.QLabel(form)
        # 位置设置为（10, 30），大小为（320, 240）
        self.picture.setGeometry(QtCore.QRect(10, 30, 320, 240))
        # 设置背景颜色为白色
        self.picture.setStyleSheet("background-color:#FFFFFF;")
        # 设置label中的文字居中
        self.picture.setAlignment(QtCore.Qt.AlignCenter)
        # 设置label的名称
        self.picture.setObjectName("picture")

        # 结果布局设置
        self.layout_widget = QtWidgets.QWidget(form)
        # 位置设置为（10, 310），大小为（320, 240）
        self.layout_widget.setGeometry(QtCore.QRect(10, 310, 320, 240))

        self.layout_widget.setObjectName("layoutWidget")
        # 布局设置为垂直布局
        self.vertical_layout = QtWidgets.QVBoxLayout(self.layout_widget)
        # 设置布局的间距为0
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setObjectName("verticalLayout")
        self.horizontal_layout = QtWidgets.QHBoxLayout()
        self.horizontal_layout.setObjectName("horizontalLayout")

        # 功能按钮
        self.pushButton_select_img = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_select_img.setObjectName("pushButton_1")
        self.vertical_layout.addWidget(self.pushButton_select_img)

        self.pushButton_camera = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_camera.setObjectName("pushButton_2")
        self.pushButton_select_img.setFont(font)
        self.vertical_layout.addWidget(self.pushButton_camera)

        self.pushButton_video = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_video.setObjectName("pushButton_3")
        self.vertical_layout.addWidget(self.pushButton_video)

        # self.vertical_layout.addLayout(self.horizontal_layout)

        # “识别结果”
        self.label_result = QtWidgets.QLabel(form)
        self.label_result.setGeometry(QtCore.QRect(360, 50, 120, 40))
        self.label_result.setObjectName("label_result")
        # 表情标签
        self.label_emotion = QtWidgets.QLabel(form)
        self.label_emotion.setGeometry(QtCore.QRect(500, 50, 120, 40))
        self.label_emotion.setObjectName("label_emotion")
        self.label_emotion.setAlignment(QtCore.Qt.AlignCenter)
        # emoji
        self.emoji = QtWidgets.QLabel(form)
        self.emoji.setGeometry(QtCore.QRect(630, 0, 200, 200))
        self.emoji.setAlignment(QtCore.Qt.AlignCenter)
        self.emoji.setObjectName("emoji")

        # “识别结果”
        self.time_label = QtWidgets.QLabel(form)
        self.time_label.setGeometry(QtCore.QRect(360, 100, 120, 40))
        self.time_label.setObjectName("time_label")
        # 运行时间
        self.time = QtWidgets.QLabel(form)
        self.time.setGeometry(QtCore.QRect(500, 100, 120, 40))
        self.time.setObjectName("time")
        self.time.setAlignment(QtCore.Qt.AlignCenter)

        self.label_bar = QtWidgets.QLabel(form)
        self.label_bar.setGeometry(QtCore.QRect(700, 150, 120, 40))
        self.label_bar.setObjectName("label_bar")
        #概率直方图
        self.graphicsView = QtWidgets.QGraphicsView(form)
        self.graphicsView.setGeometry(QtCore.QRect(360, 200, 800, 650))
        self.graphicsView.setObjectName("graphicsView")

        self.pushButton_select_img.clicked.connect(self.open_file_browser)
        self.retranslate_ui(form)
        QtCore.QMetaObject.connectSlotsByName(form)

    def retranslate_ui(self, form):
        _translate = QtCore.QCoreApplication.translate
        form.setWindowTitle(_translate("Form", "Face Emotion Recognition"))
        self.picture.setText(_translate("Form", ""))
        self.pushButton_select_img.setText(_translate("Form", "选择图像"))
        self.pushButton_select_img.setFont(font)
        self.pushButton_camera.setText(_translate("Form", "选择视频"))
        self.pushButton_camera.setFont(font)
        self.pushButton_video.setText(_translate("Form", "打开摄像头"))
        self.pushButton_video.setFont(font)
        self.label_result.setText(_translate("Form", "识别结果："))
        self.label_result.setFont(font)
        self.label_emotion.setText(_translate("Form", "表情标签"))
        self.label_emotion.setFont(font)
        self.emoji.setText(_translate("Form", "emoji"))
        self.time_label.setText(_translate("Form", "识别时间："))
        self.time_label.setFont(font)
        self.time.setFont(font)
        self.label_bar.setText(_translate("Form", "概率直方图"))
        self.label_bar.setFont(font)

    def open_file_browser(self):
        # 加载模型
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="选取图片", directory="../dataset/ck+",
                                                                     filter="All Files (*);;Text Files (*.txt)")

        # 显示原图
        if file_name is not None and file_name != "":
            self.show_raw_img(file_name)
            print(f'open_file_browser: {file_name}')
            start_time = time.time()
            emotion, possibility = predict_expression(file_name, self.model)
            end_time = time.time()
            # 将possiblity的数据归一化
            possibility = possibility / np.sum(possibility)
            print(f'open_file_browser: {emotion}, {possibility}')
            self.show_results(emotion, possibility)
        run_time = round(end_time - start_time, 2)
        self.time.setText(QtCore.QCoreApplication.translate("Form", str(run_time) + 's'))
        print(f'open_file_browser: {round(end_time - start_time, 2)}s')

    def show_raw_img(self, filename):
        img = cv2.imread(filename)
        frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320, 240))
        self.picture.setPixmap(QtGui.QPixmap.fromImage(
            QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                         QtGui.QImage.Format_RGB888)))

    def show_results(self, emotion, possibility):
        # 显示表情名
        print(f'show_results: {emotion}')
        print(f'{str(emotion)}')
        self.label_emotion.setText(QtCore.QCoreApplication.translate("Form", emotion))
        # 显示emoji
        if emotion != 'no':
            img = cv2.imread('./assets/icons/' + str(emotion) + '.png')
            print(f'show_results_img: {img}')
            frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (100, 100))
            self.emoji.setPixmap(QtGui.QPixmap.fromImage(
                QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                             QtGui.QImage.Format_RGB888)))
        else:
            self.emoji.setText(QtCore.QCoreApplication.translate("Form", "no result"))
        # 显示直方图
        self.show_bars(list(possibility))

    def show_bars(self, possbility):
        dr = MyFigureCanvas()
        dr.draw_(possbility)
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)
        self.graphicsView.setScene(graphicscene)
        self.graphicsView.show()

    def get_faces_from_image(self, img_path):
        """
        获取图片中的人脸
        :param img_path:
        :return:
        """
        import cv2
        face_cascade = cv2.CascadeClassifier('./data/params/haarcascade_frontalface_alt.xml')
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

class MyFigureCanvas(FigureCanvas):


    def __init__(self, parent=None, width=7, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)

    def draw_(self, possibility):
        x = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        self.axes.bar(x, possibility, align='center', color='c', alpha=0.8, width=0.5)
        self.axes.set_xticklabels(x,rotation=35,fontsize=12)  # 设置标签文本
