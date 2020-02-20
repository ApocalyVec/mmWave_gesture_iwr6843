import sys

import cv2
import qimage2ndarray
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QVBoxLayout, QWidget
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading
import matplotlib
import matplotlib.pyplot as plt

from realtime.simulation import sim_heatmap


class ProcessWorker(QObject):
    imageChanged = pyqtSignal(QImage)

    def doWork(self):
        while True:
            im = testColourMap()
            qim = qimage2ndarray.array2qimage(im, normalize=True)
            self.imageChanged.emit(qim)
            QThread.msleep(33)

class Widget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.resize(500, 500);

        lay = QVBoxLayout(self)
        gv = QGraphicsView()
        lay.addWidget(gv)
        scene = QGraphicsScene(self)
        gv.setScene(scene)
        self.pixmap_item = QGraphicsPixmapItem()
        scene.addItem(self.pixmap_item)

        # im = testColourMap()
        # gv.scaleToImage(im)

        self.workerThread = QThread()
        self.worker = ProcessWorker()
        self.worker.moveToThread(self.workerThread)
        self.workerThread.finished.connect(self.worker.deleteLater)
        self.workerThread.started.connect(self.worker.doWork)
        self.worker.imageChanged.connect(self.setImage)
        self.workerThread.start()


    @pyqtSlot(QImage)
    def setImage(self, image):
        qpixmap = QPixmap(image)
        self.pixmap_item.setPixmap(qpixmap)


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
    # qim = qimage2ndarray.array2qimage(color_matrix, normalize=True)
    return color_matrix


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())