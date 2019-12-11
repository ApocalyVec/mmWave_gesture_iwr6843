from keras.engine.saving import load_model
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.graph_utils import smooth_heat_scatter
from utils.path_utils import generate_train_val_ids
from sklearn.preprocessing import MinMaxScaler

regressor = load_model('D:/PycharmProjects/mmWave_gesture_iwr6843/models/thm_model.h5')
regressor.summary()
# this is step = 5
# dataset_path = 'E:/alldata_second/alldataset/thm_ts5/thm_dataset'
# label_dict_path = 'E:/alldata_second/alldataset/thm_ts5/thm_label_dict.p'
# data_without_label = pickle.load(open('E:/alldata_second/alldataset/thm_ts5/thm_data_without_label.p', 'rb'))

# this is step = 1
dataset_path = 'D:/alldataset/thm_dataset'
label_dict_path = 'D:/alldataset/thm_label_dict.p'
data_without_label = pickle.load(open('D:/alldataset/thm_data_without_label.p', 'rb'))

labels = pickle.load(open(label_dict_path, 'rb'))
scaler = pickle.load(open('D:/PycharmProjects/mmWave_gesture_iwr6843/models/scalers/120519_data_scaler.p', 'rb'))

# videoData_path = ''
# video_frame_list = os.listdir(videoData_path)
# video_frame_timestamps = list(map(lambda x: float(x.strip('.jpg')), video_frame_list))

# get the Y and X
# take only the first 100 for plotting
X = []
Y = []


for i, data in enumerate(os.listdir(dataset_path)):
    print('Loading ' + str(i) + ' of ' + str(len(os.listdir(dataset_path))))
    if data.strip('.npy') not in data_without_label:
        X.append(np.load(os.path.join(dataset_path, data)))
        Y.append(labels[data.strip('.npy')])
    # if len(X) >= 480:
    #     break

X = np.asarray(X)
Y = np.asarray(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=3, shuffle=True)


# make the prediction
print('Predicting...')
Y_predict = regressor.predict(X_test, batch_size=16)

# inverse scale
Y_predict = scaler.inverse_transform(Y_predict)

# # plot the result
# fig = plt.figure()
# fig.set_size_inches(40, 16.5)
#
# ax = plt.subplot(311)
# ax.set_title('Radar vs. CV tracking: X Axis')
# plt.plot(Y[:, 0], label='CV Predicted X Displacement (Ground Truth)')
# plt.plot(Y_predict[:, 0], label='X Displacement from Radar Tracking')
# plt.legend()
# lgd = ax.legend(loc=3)
#
# ax = plt.subplot(312)
# ax.set_title('Radar vs. CV tracking: Y Axis')
# fig.suptitle('Y Displacement')
# plt.plot(Y[:, 1], label='CV Predicted Y Displacement (Ground Truth)')
# plt.plot(Y_predict[:, 1], label='Y Displacement from Radar Tracking')
# plt.legend()
# lgd = ax.legend(loc=2)
#
# ax = plt.subplot(313)
# ax.set_title('Radar vs. CV tracking: Z Axis')
# plt.plot(Y[:, 2], label='CV Predicted Y Displacement (Ground Truth)')
# plt.plot(Y_predict[:, 1], label='Y Displacement from Radar Tracking')
# plt.legend()
# lgd = ax.legend(loc=2)
# plt.show()
#
# # plot continuous tracking
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Radar vs. CV tracking')
# start = 1
# end = 11
# numbering = [str(x) for x in list(range(1, end-start + 1))]
# plt.scatter(Y_predict[start:end, 0], Y_predict[start:end, 1], label='Radar Tracking')
# plt.scatter(Y[start:end, 0], Y[start:end, 1], label='CV Tracking (Ground Truth)')
# plt.plot(Y_predict[start:end, 0], Y_predict[start:end, 1])
# plt.plot(Y[start:end, 0], Y[start:end, 1])
# ax.set_xlabel('x displacement')
# ax.set_ylabel('y displacement')
# plt.legend()
# lgd = ax.legend(loc=10, bbox_to_anchor=(0.5,-0.12))
# last_x = 0
# last_y = 0
# for i, item in enumerate(zip(Y[start:end, 0], Y[start:end, 1],
#                              Y_predict[start:end, 0], Y_predict[start:end, 1],
#                              numbering)):
#     x = item[0]
#     y = item[1]
#     x_pred = item[2]
#     y_pred = item[3]
#     text = item[4]
#
#     if abs(last_x - x) < 0.3:
#         x = x + 0.4
#     if abs(last_y - y) < 0.3:
#         y = y + 0.4
#     last_x = x
#     last_y = y
#
#     plt.annotate(text, (x + 0.3, y + 0.3), color='orange')
#     plt.annotate(text, (x_pred + 0.3, y_pred + 0.3), color='blue')
# plt.show()

# plot the discrepancy
# fig = plt.figure()
# ax = fig.add_subplot(111)
# y_diff = Y_test - Y_predict
# plt.scatter(list(range(100)), y_diff[:, 0], s=6, label='X Difference')
# plt.scatter(list(range(100)), y_diff[:, 1], s=6, label='Y Difference')
# plt.scatter(list(range(100)), y_diff[:, 2], s=6, label='Z Difference')
# plt.legend()
# plt.show()

track_x_diff = Y_test[:, 0] - Y_predict[:, 0]
track_y_diff = Y_test[:, 1] - Y_predict[:, 1]
track_z_diff = Y_test[:, 2] - Y_predict[:, 2]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

# Create a figure with 6 plot areas
data = np.transpose(np.asarray([track_x_diff, track_y_diff]))
x = track_x_diff
y = track_y_diff

sct_size = 75
# # Create a figure with 6 plot areas
# fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))
#
# Everything sarts with a Scatterplot
# axes[0].set_title('Scatterplot')
# axes[0].plot(x, y, 'ko')
# As you can see there is a lot of overplottin here!

# Thus we can cut the plotting window in several hexbins
nbins = 100
# axes[1].set_title('Hexbin')
# axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)

# 2D Histogram
# axes[2].set_title('2D Histogram')
# axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
k = kde.gaussian_kde(data.T)
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# plot a density
# axes[3].set_title('Calculate Gaussian KDE')
# axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)

# add shading
# axes[4].set_title('2D Density with shading')
# axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

# contour
# axes[5].set_title('Contour')
# axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
# axes[5].contour(xi, yi, zi.reshape(xi.shape))

fig = plt.figure()
fig.set_size_inches(10, 6)

ax11 = plt.subplot(234)
ax11.scatter(x, y, s=6, c='orange')
ax11.set_xlim(-sct_size, sct_size)
ax11.set_ylim(-sct_size, sct_size)
ax11.set_xlabel('X (pixels)')
ax11.set_ylabel('Y (pixels)')
ax11.set_title("X-Y Variance")
ax11.grid()

ax1 = plt.subplot(231)
ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Oranges_r)
ax1.contour(xi, yi, zi.reshape(xi.shape))
ax1.set_xlim(-8, 8)
ax1.set_ylim(-8, 8)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_title("X-Y Contour Color Map")

data = np.transpose(np.asarray([track_x_diff, track_z_diff]))
x = track_x_diff
y = track_z_diff

# Everything sarts with a Scatterplot
# axes[0].set_title('Scatterplot')
# axes[0].plot(x, y, 'ko')
# As you can see there is a lot of overplottin here!

# Thus we can cut the plotting window in several hexbins
nbins = 40
# axes[1].set_title('Hexbin')
# axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)

# 2D Histogram
# axes[2].set_title('2D Histogram')
# axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
k = kde.gaussian_kde(data.T)
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# plot a density
# axes[3].set_title('Calculate Gaussian KDE')
# axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)

# add shading
# axes[4].set_title('2D Density with shading')
# axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

# contour
# axes[5].set_title('Contour')
# axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
# axes[5].contour(xi, yi, zi.reshape(xi.shape))

ax21 = plt.subplot(235)
ax21.scatter(x, y, s=6, c='orange')
ax21.set_xlabel('X (pixels)')
ax21.set_ylabel('Z (pixels)')
ax21.set_xlim(-sct_size, sct_size)
ax21.set_ylim(-sct_size, sct_size)
ax21.set_title("X-Z Variance")
ax21.grid()

ax2 = plt.subplot(232)
ax2.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Oranges_r)
ax2.contour(xi, yi, zi.reshape(xi.shape))
ax2.set_xlim(-8, 8)
ax2.set_ylim(-8, 8)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Z (pixels)')
ax2.set_title("X-Z Contour Colormap")

data = np.transpose(np.asarray([track_y_diff, track_z_diff]))
x = track_y_diff
y = track_z_diff

# Everything sarts with a Scatterplot
# axes[0].set_title('Scatterplot')
# axes[0].plot(x, y, 'ko')
# As you can see there is a lot of overplottin here!

# Thus we can cut the plotting window in several hexbins
nbins = 100
# axes[1].set_title('Hexbin')
# axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)

# 2D Histogram
# axes[2].set_title('2D Histogram')
# axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
k = kde.gaussian_kde(data.T)
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# plot a density
# axes[3].set_title('Calculate Gaussian KDE')
# axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)

# add shading
# axes[4].set_title('2D Density with shading')
# axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

# contour
# axes[5].set_title('Contour')
# axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
# axes[5].contour(xi, yi, zi.reshape(xi.shape))

ax31 = plt.subplot(236)
ax31.scatter(x, y, s=6, c='orange')
ax31.set_xlabel('Y (pixels)')
ax31.set_ylabel('Z (pixels)')
ax31.set_xlim(-sct_size, sct_size)
ax31.set_ylim(-sct_size, sct_size)
ax31.set_title("Y-Z Variance")
ax31.grid()

ax3 = plt.subplot(233)
ax3.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Oranges_r)
ax3.contour(xi, yi, zi.reshape(xi.shape))
ax3.set_xlim(-8, 8)
ax3.set_ylim(-8, 8)
ax3.set_xlabel('Y (pixels)')
ax3.set_ylabel('Z (pixels)')
ax3.set_title("Y-Z Contour Colormap")

plt.show()


from sklearn import metrics
print('MSE X is:')
print(metrics.mean_squared_error(Y_test[:, 0], Y_predict[:, 0]))

print('MSE Y is:')
print(metrics.mean_squared_error(Y_test[:, 1], Y_predict[:, 1]))

print('MSE Z is:')
print(metrics.mean_squared_error(Y_test[:, 2], Y_predict[:, 2]))

print('MSE XY is:')
print(metrics.mean_squared_error(Y_test[:, :2], Y_predict[:, :2]))

print('MSE Overall is:')
print(metrics.mean_squared_error(Y_test, Y_predict))


print('STD X is:')
print(np.std(track_x_diff))
print('STD Y is:')
print(np.std(track_y_diff))
print('STD Z is:')
print(np.std(track_z_diff))

print('STD XY is:')
print(np.std(np.asarray([track_x_diff, track_y_diff])))

print('STD Overall is:')
print(np.std(np.asarray([track_x_diff, track_y_diff, track_z_diff])))

x_ratio = 2.70
y_ratio = 4.05

print('MSE X in mm is:')
print(metrics.mean_squared_error(Y_test[:, 0]/x_ratio, Y_predict[:, 0]/x_ratio))
print('MSE Y in mm is:')
print(metrics.mean_squared_error(Y_test[:, 1]/y_ratio, Y_predict[:, 1]/y_ratio))

print('MSE XY in mm is:')
print(metrics.mean_squared_error(np.asarray([Y_test[:, 0]/x_ratio, Y_test[:, 1]/y_ratio]),
                                 np.asarray([Y_predict[:, 0]/x_ratio, Y_predict[:, 1]/y_ratio])))

print('STD XY in mm is:')
print(np.std(np.asarray([track_x_diff/x_ratio, track_y_diff/y_ratio])))