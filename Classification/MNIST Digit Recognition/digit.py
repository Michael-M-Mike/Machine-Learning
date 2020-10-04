
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import ImageGrab
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model("digit_recognition_model_CNN")


class MyGraphicsView(QtWidgets.QGraphicsView):

    def __init__(self, parent):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.pressed_flag = False

    def mouseMoveEvent(self, event):
        ui.draw(event.pos())

    def mousePressEvent(self, event):
        self.pressed_flag = True

    def mouseReleaseEvent(self, event):
        self.pressed_flag = False


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(500, 600))
        MainWindow.setMaximumSize(QtCore.QSize(500, 600))
        font = QtGui.QFont()
        font.setFamily("Gabriola")
        font.setPointSize(18)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_prediction = QtWidgets.QFrame(self.centralwidget)
        self.frame_prediction.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_prediction.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_prediction.setObjectName("frame_prediction")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_prediction)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_static = QtWidgets.QLabel(self.frame_prediction)
        self.label_static.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_static.setObjectName("label_static")
        self.horizontalLayout_2.addWidget(self.label_static)
        self.label_prediction = QtWidgets.QLabel(self.frame_prediction)
        self.label_prediction.setObjectName("label_prediction")
        self.horizontalLayout_2.addWidget(self.label_prediction)
        self.verticalLayout.addWidget(self.frame_prediction)
        self.frame_digit = QtWidgets.QFrame(self.centralwidget)
        self.frame_digit.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_digit.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_digit.setObjectName("frame_digit")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_digit)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = MyGraphicsView(self.frame_digit)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame_digit)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.b_predict = QtWidgets.QPushButton(self.centralwidget)
        self.b_predict.setObjectName("b_predict")
        self.horizontalLayout.addWidget(self.b_predict)
        self.b_clear = QtWidgets.QPushButton(self.centralwidget)
        self.b_clear.setObjectName("b_clear")
        self.horizontalLayout.addWidget(self.b_clear)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        self.b_clear.clicked.connect(self.clear)
        self.b_predict.clicked.connect(self.predict)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def draw(self, mouse_pos):
        if self.graphicsView.pressed_flag:
            grey = QtGui.QColor(QtGui.qRgb(0, 0, 0))
            pen = QtGui.QPen(grey)
            brush = QtGui.QBrush(grey)

            x = mouse_pos.x()
            y = mouse_pos.y()
            self.scene.setSceneRect(0, 0, self.graphicsView.width() - 5, self.graphicsView.height() - 5)
            self.scene.addEllipse(x, y, 30, 30, pen, brush)

    def clear(self):
        self.scene.clear()
        self.label_prediction.setText("...")

    def predict(self):

        left = MainWindow.geometry().left() + self.graphicsView.x() + 20
        upper = MainWindow.geometry().top() + self.graphicsView.y() + 90
        right = left + self.graphicsView.geometry().width() - 30
        lower = upper + self.graphicsView.geometry().height() - 20

        img = ImageGrab.grab(bbox=(left, upper, right, lower))
        img = img.resize((28, 28))
        img = np.array(img.convert("L"))
        img = np.invert(img)

        x_test = np.array(img)
        x_test = x_test / 255
        x_test = x_test.reshape(1, 28, 28, 1)
        prediction = model.predict(x_test)

        prediction = np.argmax(prediction[0])
        self.label_prediction.setText(str(prediction))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Digit Recognition App"))
        self.label_static.setText(_translate("MainWindow", "Prediction:"))
        self.label_prediction.setText(_translate("MainWindow", "..."))
        self.b_predict.setText(_translate("MainWindow", "Predict"))
        self.b_clear.setText(_translate("MainWindow", "Clear"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
