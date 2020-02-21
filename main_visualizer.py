import sys

import cv2
import qimage2ndarray
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QVBoxLayout, QWidget, \
    QGridLayout, QMainWindow, qApp, QLabel
import time
import pyqtgraph as pg
from realtime.simulation import sim_heatmap, sim_detected_points
from utils.img_utils import array_to_colormap_qim, array_to_3D_scatter_qim


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
        spec_array = sim_heatmap((100, 100))
        spec_qim = array_to_colormap_qim(spec_array)

        pts_array = sim_detected_points()
        # pts_qim = array_to_3D_scatter_qim(pts_array)

        self.signals.result.emit({'spec': spec_qim,
                                  'pts': pts_array})


class MainWindow(QMainWindow):
    def __init__(self, refresh, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.resize(1920, 1080)
        pg.setConfigOption('background', 'w')

        w = QWidget()
        # create main layout
        self.lay = QGridLayout(self)
        # add spectrogram graphic view
        self.spec_pixmap_item = QGraphicsPixmapItem()
        self.init_spec_view()
        # add detected points plots
        self.scatterXY = self.init_pts_view(pos=(0, 3))
        self.scatterZD = self.init_pts_view(pos=(0, 4))

        # add the interrupt button
        self.interruptBtn = QtWidgets.QPushButton(text='Interrupt')
        self.interruptBtn.clicked.connect(self.interruptBtnAction)
        self.lay.addWidget(self.interruptBtn, *(0, 1))

        # add dialogue label
        self.dialogueLabel = QLabel()
        self.dialogueLabel.setText("Running")
        self.lay.addWidget(self.dialogueLabel, *(0, 0))

        # set the main layout
        w.setLayout(self.lay)
        self.setCentralWidget(w)
        self.show()

        # create thread pool
        self.threadpool = QThreadPool()
        self.timer = QTimer()
        self.timer.setInterval(refresh)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def init_spec_view(self):
        spc_gv = QGraphicsView()
        self.lay.addWidget(spc_gv, *(0, 2))
        scene = QGraphicsScene(self)
        spc_gv.setScene(scene)
        scene.addItem(self.spec_pixmap_item)

    def init_pts_view(self, pos):

        pts_plt = pg.PlotWidget()
        pts_plt.setXRange(0., 1.)
        pts_plt.setYRange(0., 1.)
        self.lay.addWidget(pts_plt, *pos)
        scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='b'), symbol='o', size=1)
        pts_plt.addItem(scatter)
        return scatter
        # pts_gv = QGraphicsView()
        # self.lay.addWidget(pts_gv, *(0, 3))
        # scene = QGraphicsScene(self)
        # pts_gv.setScene(scene)
        # scene.addItem(self.pts_pixmap_item)

    def interruptBtnAction(self):
        self.timer.stop()
        self.dialogueLabel.setText('Stopped. Close the application to return to the console.')

    def recurring_timer(self):
        worker = Worker()
        worker.signals.result.connect(self.update_image)
        self.threadpool.start(worker)

    def update_image(self, data_dict):
        spec_qpixmap = QPixmap(data_dict['spec'])
        self.spec_pixmap_item.setPixmap(spec_qpixmap)
        # update the scatter
        self.scatterXY.setData(data_dict['pts'][:, 0], data_dict['pts'][:, 1])
        self.scatterZD.setData(data_dict['pts'][:, 2], data_dict['pts'][:, 3])


if __name__ == '__main__':
    refresh = 33  # refresh every x ms

    app = QApplication(sys.argv)
    window = MainWindow(refresh=refresh)
    app.exec_()
