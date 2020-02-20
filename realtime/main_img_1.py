import sys

import qimage2ndarray as qimage2ndarray
from PyQt5 import QtWidgets, QtGui
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt


class Example(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.initUI()

    def initUI(self):
        graphicsView = QtWidgets.QGraphicsView(self)
        scene = QtWidgets.QGraphicsScene()
        self.pixmap = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self.pixmap)
        graphicsView.setScene(scene)

        self.setGeometry(300, 300, 300, 200)
        graphicsView.resize(300,200)
        self.setWindowTitle('Test')
        self.show()

    def play(self):
        img = testColourMap()
        self.pixmap.setPixmap(img)
        print('Updated')


def testColourMap():
    # sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
    # fig = Figure((2.5,.2), subplotpars = sp)
    # canvas = FigureCanvas(fig)
    # ax = fig.add_subplot(111)
    # gradient = np.linspace(0, 1, 256)
    # gradient = np.vstack((gradient, gradient))
    # ax.imshow(gradient, aspect=10, cmap=cmap)
    # ax.set_axis_off()
    # canvas.draw()
    # size = canvas.size()
    # width, height = size.width(), size.height()
    im = plt.imshow(np.random.random((100, 100)))
    color_matrix = im.cmap(im.norm(im.get_array()))
    qim = qimage2ndarray.array2qimage(color_matrix, normalize=True)
    return QPixmap(qim)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Example()
    window.play()
    app.exec()
    sys.exit()