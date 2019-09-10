import datetime
import time

from iwr6843_utils import serial_iwr6843
import collections
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from threading import Thread, Event
import matplotlib.pyplot as plt
import os
import pickle

# data queue global
from utils.data_utils import produce_voxel, StreamingMovingAverage
import numpy as np
import pyautogui


data_q = collections.deque(maxlen=1)
data_list = []
processed_data_list = []
data_shape = (1, 25, 25, 25)

# set up graph
# START QtAPP for the plot
app = QtGui.QApplication([])
thm_gui_size = 640, 480

# Set the xy plot
pg.setConfigOption('background', 'w')
win = pg.GraphicsWindow(title="2D scatter plot")
fig_z_y = win.addPlot()
fig_z_y.setXRange(-0.5, 0.5)
fig_z_y.setYRange(0, 1.5)
fig_z_y.setLabel('left', text='Y position (m)')
fig_z_y.setLabel('bottom', text='X position (m)')
xy_graph = fig_z_y.plot([], [], pen=None, symbol='o')

# set the zv plot
fig_z_v = win.addPlot()
fig_z_v.setXRange(-1, 1)
fig_z_v.setYRange(-1, 1)
fig_z_v.setLabel('left', text='Z position (m)')
fig_z_v.setLabel('bottom', text='Doppler (m/s)')
zd_graph = fig_z_v.plot([], [], pen=None, symbol='o')

# set the thumouse plot
fig_thumouse = win.addPlot()
fig_thumouse.setXRange(0, thm_gui_size[0])
fig_thumouse.setYRange(0, thm_gui_size[1])
thumouse_graph = fig_thumouse.plot([], [], pen=None, symbol='o')

# thread variables
main_stop_flag = False

today = datetime.datetime.now()
root_dn = 'data/f_data-' + str(today).replace(':', '-').replace(' ', '_')

# Model Globals
is_simulate = False
is_predict = True


if is_predict:
    from utils.model_wrapper import NeuralNetwork, onehot_decoder


class InputThread(Thread):
    def __init__(self, thread_id):
        Thread.__init__(self)
        self.thread_id = thread_id

    def run(self):
        global is_collecting_started

        input()
        is_collecting_started = True


class PredictionThread(Thread):
    def __init__(self, thread_id, model_encoder_dict, timestep, thumouse_gui=None, mode=None):

        Thread.__init__(self)
        self.thread_id = thread_id
        self.model_encoder_dict = model_encoder_dict
        self.timestep = timestep
        # create a sequence buffer of shape: timestemp * shape of the data
        self.mode = mode
        if 'thm' in mode:
            self.thumouse_gui = thumouse_gui

        if 'idp' in mode:
            pass

    def run(self):
        global main_stop_flag
        global thm_gui_size
        global data_shape
        global is_point

        # general vars
        buffer_size = self.timestep
        sequence_buffer = np.zeros(tuple([buffer_size] + list(data_shape)))

        # thumouse related vars
        mouse_x = 0.0
        mouse_y = 0.0
        x_factor = 10.0
        y_factor = 0

        ma_x = StreamingMovingAverage(window_size=15)
        ma_y = StreamingMovingAverage(window_size=100)

        if 'thm' in self.mode:
            thm_model = self.model_encoder_dict['thm'][0]
            thm_decoder = self.model_encoder_dict['thm'][1]
            gui_wid_hei = thm_gui_size

        # idp related vars
        if 'idp' in self.mode:
            idp_model = self.model_encoder_dict['idp'][0]
            idp_threshold = 0.75
            sequence_buffer = np.zeros(tuple([buffer_size] + list(data_shape)))
            idp_pred_dict = {0: 'A', 1: 'D', 2: 'L', 3: 'M', 4: 'P', 5: 'nothing'}

        while True:
            time.sleep(0.2)
            # retrieve the data from deque
            if main_stop_flag:
                break

            if len(data_q) != 0:
                # ditch the tail, append to head
                sequence_buffer = np.concatenate((sequence_buffer[1:], np.expand_dims(data_q.pop(), axis=0)))

                if 'idp' in self.mode:
                    pass
                    # time.sleep(0.5)
                    # idp_pre_result = idp_model.predict(x=sequence_buffer)
                    # pre_argmax = np.argmax(idp_pre_result)
                    # pre_amax = np.amax(idp_pre_result)
                    #
                    # if pre_amax > idp_threshold:  # a character is written
                    #     if pre_argmax == 5:
                    #         print('No One is Writing' + '    amax = ' + str(pre_amax))
                    #     else:
                    #         print('You just wrote: ' + idp_pred_dict[int(pre_argmax)] + '    amax = ' + str(pre_amax))
                    #         # clear the buffer
                    #         sequence_buffer = np.zeros(tuple([buffer_size] + list(data_shape)))
                    # else:
                    #     print('No writing, amax = ' + str(pre_amax))

                if 'thm' in self.mode:
                    # if not np.any(sequence_buffer[-1]):
                    thm_pred_result = thm_model.predict(x=np.expand_dims(sequence_buffer[-1], axis=0))
                    # expand dim for single sample batch
                    decoded_result = thm_decoder.inverse_transform(thm_pred_result)

                    delta_x = decoded_result[0][0] * x_factor
                    delta_y = decoded_result[0][1] * y_factor

                    delta_x_ma = ma_x.process(delta_x)
                    delta_y_ma = ma_y.process(delta_y)

                    # mouse_x = min(max(mouse_x + delta_x, 0), gui_wid_hei[0])
                    # mouse_y = min(max(mouse_y + delta_y, 0), gui_wid_hei[1])

                    # move the actual mouse
                    pyautogui.moveRel(delta_x_ma, delta_y_ma, duration=.1)
                    # if self.thumouse_gui is not None:
                        # self.thumouse_gui.setData([mouse_x], [mouse_y])

                    print(str(delta_x_ma) + ' ' + str(delta_y_ma) + '     ' + str(len(data_q)))


def load_model(model_path, encoder=None):
    model = NeuralNetwork()
    model.load(file_name=model_path)

    if encoder is not None:
        if type(encoder) == str:
            encoder = pickle.load(open(encoder, 'rb'))
            return model, encoder
    else:
        return model


def main():
    global main_stop_event
    global thumouse_graph
    global is_predict
    global main_stop_flag
    global is_point

    if is_predict:
        timestep = 100
        # my_mode = ['thm', 'idp']
        my_mode = ['thm']

        thm_model_path = 'D:/PycharmProjects/mmWave_gesture_iwr6843/models/thm_model.h5'
        thm_scaler_path = 'D:/PycharmProjects/mmWave_gesture_iwr6843/models/scalers/thm_scaler.p'
        # idp_model_path = 'D:/code/DoubleMU/models/palmPad_model.h5'

        model_dict = {'thm': load_model(thm_model_path,
                                        encoder=thm_scaler_path),
                      # 'idp': load_model(idp_model_path,
                      #                   encoder=onehot_decoder())
                      }
        pred_thread = PredictionThread(1, model_encoder_dict=model_dict,
                                       timestep=timestep, thumouse_gui=thumouse_graph, mode=my_mode)
        pred_thread.start()

    # start input thread
    # input_thread = InputThread(1)
    # input_thread.start()

    configFileName = 'profiles/profile_further_tuned.cfg'
    dataPortName = 'COM9'
    userPortName = 'COM8'

    # open the serial port to the radar
    user_port, data_port = serial_iwr6843.serialConfig(configFileName, dataPortName=dataPortName,
                                                       userPortName=userPortName)
    serial_iwr6843.clear_serial_buffer(user_port, data_port)
    # give some time for the board to boot
    time.sleep(2)

    serial_iwr6843.sensor_start(user_port)
    time.sleep(2)

    input('Press Enter to Start...')
    serial_iwr6843.clear_serial_buffer(user_port, data_port)
    print('Started! Press CTRL+C to interrupt...')

    while True:
        try:
            detected_points = serial_iwr6843.parse_stream(data_port)

            if detected_points is not None:
                frame_timestamp = time.time()
                processed_data = produce_voxel(detected_points)
                data_list.append((frame_timestamp, detected_points))
                processed_data_list.append((frame_timestamp, processed_data))

                if is_predict and len(detected_points) != 0:
                    data_q.append(processed_data)

                xy_graph.setData(detected_points[:, 0], detected_points[:, 1])
                # zd_graph.setData(detected_points[:, 2], detected_points[:, 3])

                zd_graph.setData(detected_points[:, 2], detected_points[:, 3])

            else:
                pass
                # print('Packet is not complete yet!')

            QtGui.QApplication.processEvents()
        except KeyboardInterrupt as ki:
            break

    time.sleep(1)

    # close the connection to the sensor
    print('Sending Stop Command')
    serial_iwr6843.sensor_stop(user_port)
    serial_iwr6843.close_connection(user_port, data_port)
    # close qtgui window
    win.close()

    # print the information about the frames collected
    print('The number of frame collected is ' + str(len(data_list)))
    time_record = max(x[0] for x in data_list) - min(x[0] for x in data_list)
    expected_frame_num = time_record * 20
    frame_drop_rate = len(data_list) / expected_frame_num
    print('Recording time is ' + str(time_record))
    print('The expected frame num is ' + str(expected_frame_num))
    print('Frame drop rate is ' + str(1 - frame_drop_rate))

    # close all the threads
    main_stop_flag = True
    if is_predict:
        pred_thread.join()

    # do you wish to save the recorded frames?
    is_save = input('do you wish to save the recorded frames? [y/n]')

    if is_save == 'y':
        os.mkdir(root_dn)
        # save the points file
        point_file_path = os.path.join(root_dn, 'f_data_points.p')
        with open(point_file_path, 'wb') as pickle_file:
            pickle.dump(data_list, pickle_file)

        # save the processed file
        voxel_file_path = os.path.join(root_dn, 'f_data_voxel.p')
        with open(voxel_file_path, 'wb') as pickle_file:
            pickle.dump(processed_data_list, pickle_file)

    else:
        print('exit without saving')


if __name__ == '__main__':
    main()
    print('Finished!')
