import sys

import cv2
import qimage2ndarray
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QVBoxLayout, QWidget, \
    QGridLayout, QMainWindow, qApp, QLabel
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading
import matplotlib
import matplotlib.pyplot as plt

from realtime.simulation import sim_heatmap


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__()
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        frame_hm_im = sim_heatmap((100, 100))
        frame_hm_qim = image_to_qim(frame_hm_im)
        self.signals.result.emit(frame_hm_qim)


class MainWindow(QMainWindow):
    def __init__(self, refresh, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.resize(1280, 720)
        w = QWidget()

        # create main layout
        lay = QGridLayout(self)
        # add graphic view
        gv = QGraphicsView()
        lay.addWidget(gv, *(0, 2))
        scene = QGraphicsScene(self)
        gv.setScene(scene)
        self.pixmap_item = QGraphicsPixmapItem()
        scene.addItem(self.pixmap_item)
        # add the interrupt buttong
        self.interruptBtn = QtWidgets.QPushButton(text='Interrupt')
        self.interruptBtn.clicked.connect(self.interruptBtnAction)
        lay.addWidget(self.interruptBtn, *(0, 1))
        # add dialogue label
        self.dialogueLabel = QLabel()
        self.dialogueLabel.setText("Running")
        lay.addWidget(self.dialogueLabel, *(0, 0))

        w.setLayout(lay)
        self.setCentralWidget(w)
        self.show()

        # create thread pool
        self.threadpool = QThreadPool()
        self.timer = QTimer()
        self.timer.setInterval(refresh)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

        # self.workerThread = QThread()
        # self.worker = ProcessWorker()
        # self.worker.moveToThread(self.workerThread)
        # self.workerThread.finished.connect(self.worker.deleteLater)
        # self.workerThread.started.connect(self.worker.doWork)
        # self.worker.imageChanged.connect(self.setImage)
        # self.workerThread.start()

    def interruptBtnAction(self):
        self.timer.stop()
        self.dialogueLabel.setText('Stopped. Close the application to return to the console.')

    def recurring_timer(self):
        worker = Worker()
        worker.signals.result.connect(self.update_image)
        self.threadpool.start(worker)

    def update_image(self, image):
        qpixmap = QPixmap(image)
        self.pixmap_item.setPixmap(qpixmap)


timing_list = []


def image_to_qim(im):
    start = time.time()
    im = plt.imshow(im)
    color_matrix = im.cmap(im.norm(im.get_array()))
    qim = qimage2ndarray.array2qimage(color_matrix, normalize=True)
    timing_list.append(time.time() - start)
    return qim


if __name__ == '__main__':
    refresh = 33

    app = QApplication(sys.argv)
    window = MainWindow(refresh=refresh)
    app.exec_()
