import time

from iwr6843_utils import serial_iwr6843
import collections
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import threading
import matplotlib.pyplot as plt

# data queue global
from utils.data_utils import preprocess_frame

data_q = collections.deque(maxlen=100)
data_list = []
processed_data_list = []

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


class ProcessingThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global global_stop_flag
        global processed_data_list

        while not global_stop_flag:
            if len(data_q) != 0:
                data = data_q.pop()
                processed_data_list.append(preprocess_frame(data))


def main():
    global global_stop_flag

    configFileName = 'profiles/profile_tuned.cfg'
    dataPortName = 'COM9'
    userPortName = 'COM8'

    # processing_thread = ProcessingThread()
    # processing_thread.start()

    # open the serial port to the radar
    user_port, data_port = serial_iwr6843.serialConfig(configFileName, dataPortName=dataPortName, userPortName=userPortName)
    input('Press Enter to Start...')
    while 1:
        try:
            detected_points = serial_iwr6843.parse_stream(data_port)

            if detected_points is not None:
                frame_timestamp = time.time()
                data_q.append(detected_points)
                data_list.append((frame_timestamp, detected_points))
                processed_data_list.append((frame_timestamp, preprocess_frame(detected_points)))

                xy_graph.setData(detected_points[:, 0], detected_points[:, 1])
                zd_graph.setData(detected_points[:, 2], detected_points[:, 3])
            else:
                pass
                # print('Packet is not complete yet!')

            QtGui.QApplication.processEvents()
        except KeyboardInterrupt as es:
            time.sleep(1)

            global_stop_flag = True
            print('Sending Stop Command')
            # close the connection to the sensor
            serial_iwr6843.sensor_stop(user_port)
            serial_iwr6843.close_connection(user_port, data_port)
            # close qtgui window
            win.close()

            # wait for the threads to join
            # processing_thread.join()

            # print the information about the frames collected
            print('The number of frame collected is ' + str(len(data_list)))
            time_record = max(x[0] for x in data_list) - min(x[0] for x in data_list)
            expected_frame_num = time_record * 15
            frame_drop_rate = len(data_list) / expected_frame_num
            print('Recording time is ' + str(time_record))
            print('The expected frame num is ' + str(expected_frame_num))
            print('Frame drop rate is ' + str(1 - frame_drop_rate))

            # do you wish to save the recorded frames?
            # is_save = input('do you wish to save the recorded frames? [y/n]')
            #
            # if is_save == 'y':
            #     os.mkdir(root_dn)
            #     file_path = os.path.join(root_dn, 'f_data.p')
            #     with open(file_path, 'wb') as pickle_file:
            #         pickle.dump(frameData, pickle_file)
            # else:
            #     print('exit without saving')
            break


if __name__ == '__main__':
    main()
    print('The end!')

