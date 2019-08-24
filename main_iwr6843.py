import time
import serial_iwr6843
import collections
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import threading

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

# data queue global
data_q = collections.deque(maxlen=100)
# data_list = []

# set up graph
# START QtAPPfor the plot
app = QtGui.QApplication([])

# Set the plot
pg.setConfigOption('background', 'w')
win = pg.GraphicsWindow(title="2D scatter plot")
fig_z_y = win.addPlot()
fig_z_y.setXRange(-0.5, 0.5)
fig_z_y.setYRange(0, 1.5)
fig_z_y.setLabel('left', text='Y position (m)')
fig_z_y.setLabel('bottom', text='X position (m)')
xy_graph = fig_z_y.plot([], [], pen=None, symbol='o')

# set the processed plot
fig_z_v = win.addPlot()
fig_z_v.setXRange(-1, 1)
fig_z_v.setYRange(-1, 1)
fig_z_v.setLabel('left', text='Z position (m)')
fig_z_v.setLabel('bottom', text='Doppler (m/s)')
zd_graph = fig_z_v.plot([], [], pen=None, symbol='o')

# thread variables
global_stop_flag = False


class PlottingThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global global_stop_flag
        global xy_graph
        global zd_graph

        while not global_stop_flag:
            if len(data_q) > 0:
                data = data_q.pop()
                xy_graph.setData(data[:, 0], data[:, 1])
                zd_graph.setData(data[:, 2], data[:, 3])
            time.sleep(0.033)

def main():
    global global_stop_flag

    configFileName = 'profile.cfg'
    dataPortName = 'COM9'
    userPortName = 'COM8'

    # open the serial port to the radar
    user_port, data_port = serial_iwr6843.serialConfig(configFileName, dataPortName=dataPortName, userPortName=userPortName)

    # start threads
    thread2 = PlottingThread()
    thread2.start()

    while 1:
        try:
            detected_points = serial_iwr6843.parse_stream(data_port)
            if detected_points is not None:
                data_q.append(detected_points)
                # data_list.append((time.time(), detected_points))

            QtGui.QApplication.processEvents()
        except KeyboardInterrupt:
            global_stop_flag = True
            print('Sending Stop Command')
            serial_iwr6843.sensor_stop(user_port)
            win.close()

            thread2.join()

            break


if __name__ == '__main__':
    main()
    print('The end!')

