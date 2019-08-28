import random
from itertools import product

import numpy as np
import math

import pickle
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN

from scipy.spatial import distance
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from utils.data_utils import produce_voxel
from utils.transformation import translate, sphere_search, rotateZ, rotateY, rotateX, scale


augmentation_list = ['trans', 'rot', 'scale', 'clipping']

# variables used in snapPointsToVolume
xmin, xmax = -0.5, 0.5
ymin, ymax = 0.0, 0.5
zmin, zmax = -0.5, 0.5

heatMin, heatMax = -1.0, 1.0
xyzScaler = MinMaxScaler().fit(np.array([[xmin, ymin, zmin],
                                         [xmax, ymax, zmax]]))
heatScaler = MinMaxScaler().fit(np.array([[heatMin],
                                          [heatMax]]))

# volumn.shape = (5, 5, 5)
def snapPointsToVolume(points, volume_shape, isClipping=False, radius=3, decay=0.8):
    """
    make sure volume is a square
    :param points: n * 4 array
    :param heat: scale 0 to 1
    :param volume:
    """
    assert len(volume_shape) == 3 and volume_shape[0] == volume_shape[1] == volume_shape[2]
    volume = np.zeros(volume_shape)

    if len(points) != 0:

        # filter out points that are outside the bounding box
        # using ABSOLUTE normalization

        points_filtered = []
        for p in points:
            if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax and zmin <= p[2] <= zmax:
                points_filtered.append(p)
        if len(points_filtered) == 0:
            return volume
        points_filtered = np.asarray(points_filtered)

        points_filtered[:, :3] = xyzScaler.transform(points_filtered[:, :3])
        points_filtered[:, 3:] = heatScaler.transform(points_filtered[:, 3:])

        size = volume_shape[0]  # the length of thesquare side
        axis = np.array((size - 1) * points_filtered[:, :3], dtype=int)  # size minus 1 for index starts at 0

        for i, row in enumerate(points_filtered):
            heat = row[3]

            volume[axis[i][0], axis[i][1], axis[i][2]] = volume[axis[i][0], axis[i][1], axis[i][2]] + heat

            if isClipping:
                point_to_clip = sphere_search(shape=volume_shape, index=(axis[i][0], axis[i][1], axis[i][2]), r=radius)
                for dist, ptc in point_to_clip:
                    if dist != 0.0:
                        factor = (radius - dist + 1) * decay /radius
                        volume[ptc[0], ptc[1], ptc[2]] = volume[ptc[0], ptc[1], ptc[2]] + heat * factor
    return volume


def radar_data_grapher_volumned(paths, is_plot=False, augmentation=(),
                                seeds=np.random.normal(0, 0.02, 5000), util_path='E:/temp'):
    # utility directory to save the pyplots
    radar_points_data_path, radar_voxel_data_path, videoData_path, figure_path, out_path, identity_string = paths

    radar_points = list(pickle.load(open(radar_points_data_path, 'rb')).items())
    radar_voxel = list(pickle.load(open(radar_voxel_data_path, 'rb')).items())

    video_frame_list = os.listdir(videoData_path)
    video_frame_timestamps = list(map(lambda x: float(x.strip('.jpg')), video_frame_list))

    style.use('fivethirtyeight')
    white_color = 'rgb(255, 255, 255)'
    black_color = 'rgb(0, 0, 0)'
    red_color = 'rgb(255, 0, 0)'

    DBSCAN_esp = 0.2
    DBSCAN_minSamples = 3

    # input data for the classifier that has the shape n*4*100, n being the number of samples

    fnt = ImageFont.truetype("arial.ttf", 16)

    # Retrieve the first timestamp
    assert [x[0] for x in radar_points] == [x[0] for x in radar_voxel]

    starting_timestamp = radar_points[0][0]
    interval_index = 1

    # removed and recreate the merged image folder
    if is_plot:
        if os.path.isdir(figure_path):
            shutil.rmtree(figure_path)
        os.mkdir(figure_path)

    volume_shape = (25, 25, 25)

    interval_volume_list = []
    this_voxel_list = []

    interval_duration = 3.0
    sample_per_sec = 15
    sample_per_interval = int(interval_duration * sample_per_sec)

    aug_string = ''
    if augmentation:
        print('Use augmentation: ' + str(augmentation))
        for aug in augmentation:
            aug_string += '_' + aug
    else:
        print('No augmentation applied')

    print('Label Cheat-sheet:')
    print('0 for DEL')
    print('1 for D')
    print('2 for E')
    print('3 for H')
    print('4 for L')
    print('5 for O')
    print('6 for R')
    print('7 for W')
    print('8 for SPC')
    print('9 for EXC')

    label_array = []
    this_label = 0

    for i, (this_points_and_ts, this_voxel_and_ts) in enumerate(zip(radar_points, radar_voxel)):

        # retrieve the timestamp making sure the data is synced
        assert this_points_and_ts[0] == this_voxel_and_ts[0]
        this_timestamp = this_points_and_ts[0]
        this_points = this_points_and_ts[1]
        this_voxel = this_voxel_and_ts[1]

        print('Processing ' + str(i + 1) + ' of ' + str(len(radar_points)) + ', interval = ' + str(interval_index))

        if is_plot:
            figure_intervaled_path = os.path.join(figure_path, str(interval_index - 1))

            if not os.path.isdir(figure_intervaled_path):
                os.mkdir(figure_intervaled_path)

            closest_video_timestamp = min(video_frame_timestamps,
                                          key=lambda x: abs(x - this_timestamp))
            closest_video_path = os.path.join(videoData_path, str(closest_video_timestamp) + '.jpg')
            closest_video_img = Image.open(closest_video_path)

            # plot the radar scatter
            ax1 = plt.subplot(2, 2, 1, projection='3d')
            ax1.set_xlim((-0.3, 0.3))
            ax1.set_ylim((-0.3, 0.3))
            ax1.set_zlim((-0.3, 0.3))
            ax1.set_xlabel('X', fontsize=10)
            ax1.set_ylabel('Y', fontsize=10)
            ax1.set_zlabel('Z', fontsize=10)
            ax1.set_title('Detected Points', fontsize=10)
            # plot the detected points
            ax1.scatter(this_points['x'], this_points['y'], this_points['z'], c=this_points['doppler'], marker='o')

        data = np.asarray([this_points['x'], this_points['y'], this_points['z'], this_points['doppler']]).transpose()
        # Do DBSCAN cluster ###########################################
        # map the points to their doppler value, this is for retrieving the doppler value after clustering

        assert produce_voxel(this_points) == this_voxel

        # apply augmentation to hand cluster #############################
        if 'trans' in augmentation:
            for p in np.nditer(this_points[:, :3], op_flags=['readwrite']):
                p[...] = p + random.choice(seeds)
        if 'rot' in augmentation:
            this_points[:, :3] = rotateX(this_points[:, :3], 720 * random.choice(seeds))
            this_points[:, :3] = rotateY(this_points[:, :3], 720 * random.choice(seeds))
            this_points[:, :3] = rotateZ(this_points[:, :3], 720 * random.choice(seeds))
        if 'scale' in augmentation:
            s = 1 + random.choice(seeds)
            this_points[:, :3] = scale(this_points[:, :3], x=s, y=s, z=s)

        if is_plot:
            ax3 = plt.subplot(2, 2, 3, projection='3d')
            ax3.set_xlim((-0.3, 0.3))
            ax3.set_ylim((-0.3, 0.3))
            ax3.set_zlim((-0.3, 0.3))
            ax3.set_xlabel('X', fontsize=10)
            ax3.set_ylabel('Y', fontsize=10)
            ax3.set_zlabel('Z', fontsize=10)
            ax3.set_title('Hand Cluster', fontsize=10)

            ax3.scatter(this_points[:, 0], this_points[:, 1], this_points[:, 2], 'o', c=this_points[:, 3], s=28,
                        marker='o')

        # create 3D feature space #############################

        this_voxel_list.append(np.expand_dims(produce_voxel(this_points), axis=0))

        # Plot the hand cluster #########################################
        # Combine the three images
        if is_plot:
            plt.savefig(os.path.join(util_path, str(this_timestamp) + '.jpg'))
            radar_3dscatter_img = Image.open(os.path.join(util_path, str(this_timestamp) + '.jpg'))

            images = [closest_video_img, radar_3dscatter_img]  # add image here to arrange them horizontally
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]

            timestamp_difference = abs(float(this_timestamp) - float(closest_video_timestamp))
            draw = ImageDraw.Draw(new_im)

            # draw the timestamp difference on the image
            (x, y) = (20, 10)
            message = "Timestamp Difference, abs(rt-vt): " + str(timestamp_difference)
            draw.text((x, y), message, fill=white_color, font=fnt)
            # draw the timestamp
            (x, y) = (20, 30)
            message = "Timestamp: " + str(this_timestamp)
            draw.text((x, y), message, fill=white_color, font=fnt)

            # draw the number of points
            (x, y) = (20, 60)
            message = "Number of detected points: " + str(this_points.shape[0])
            draw.text((x, y), message, fill=white_color, font=fnt)

            # save the combined image
            new_im.save(
                os.path.join(figure_intervaled_path, str(this_timestamp) + '_' + str(this_timestamp.as_integer_ratio()[0]) +
                             '_' + str(this_timestamp.as_integer_ratio()[1]) + '_' + str(interval_index) + '.jpg'))
            plt.close('all')

        # calculate the interval ############################
        if (this_timestamp - starting_timestamp) >= interval_duration or i == len(radar_voxel)-1:

            # decide the label
            inter_arg = 20
            if interval_index % inter_arg == 1 or interval_index % inter_arg == 2:
                this_label = 0  # for label DEL
            elif interval_index % inter_arg == 3 or interval_index % inter_arg == 4:
                this_label = 1  # for label D
            elif interval_index % inter_arg == 5 or interval_index % inter_arg == 6:
                this_label = 2  # for label E
            elif interval_index % inter_arg == 7 or interval_index % inter_arg == 8:
                this_label = 3  # for label H
            elif interval_index % inter_arg == 9 or interval_index % inter_arg == 10:
                this_label = 4  # for label L
            elif interval_index % inter_arg == 11 or interval_index % inter_arg == 12:
                this_label = 5  # for label O
            elif interval_index % inter_arg == 13 or interval_index % inter_arg == 14:
                this_label = 6  # for label R
            elif interval_index % inter_arg == 15 or interval_index % inter_arg == 16:
                this_label = 7  # for label W
            elif interval_index % inter_arg == 17 or interval_index % inter_arg == 18:
                this_label = 8  # for label SPC
            elif interval_index % inter_arg == 19 or interval_index % inter_arg == 0:
                this_label = 9  # for label EXC
            label_array.append(this_label)

            print('Interval' + str(interval_index) + ': Label-' + str(this_label) + ' # of Samples- ' + str(len(this_voxel_list)))
            print('')

            # add padding, pre-padded
            if len(this_voxel_list) < sample_per_interval:
                while len(this_voxel_list) < sample_per_interval:
                    this_voxel_list.insert(0, np.expand_dims(np.zeros(volume_shape), axis=0))
            elif len(this_voxel_list) > sample_per_interval:  # we take only the 75 most recent
                this_voxel_list = this_voxel_list[-75:]
            this_voxel_list = np.asarray(this_voxel_list)
            interval_volume_list.append(this_voxel_list)
            this_voxel_list = []
            # increment the timestamp and interval index
            starting_timestamp = starting_timestamp + interval_duration
            interval_index = interval_index + 1
        # end of end of interval processing

    # start of post processing ##########################################################################
    label_array = np.asarray(label_array)
    interval_volume_array = np.asarray(interval_volume_list)

    # validate the output shapes
    assert interval_volume_array.shape == (60, 100, 1) + volume_shape
    assert len(label_array) == 60

    print('Saving csv and npy to ' + out_path + '...')
    
    dataset_path = 'E:/indexPen/dataset'
    label_dict_path = 'E:/indexPen/labels/label_dict.p'
    # load label dict
    if os.path.exists(label_dict_path):
        label_dict = pickle.load(open(label_dict_path, 'rb'))
    else:  # create anew if does not exist
        label_dict = {}

    # put the label into the dict
    for l_index, l in enumerate(label_array):
        label_dict[identity_string + str(l_index) + aug_string] = l
    # save label dict to disk
    pickle.dump(label_dict, open(label_dict_path, 'wb'))

    # save the data chunks (intervaled volumns)
    for d_index, d in enumerate(interval_volume_array):
        print('Saving chunk #' + str(d_index))
        np.save(os.path.join(dataset_path, identity_string + str(d_index) + aug_string), d)


    print('Done saving to ' + out_path)


def generate_path(subject_name: str, case_index: int, mode: str) -> tuple:

    identity_string = subject_name + '_' + str(case_index)
    f_dir = 'f_data_' + mode + '_' + identity_string
    v_dir = 'v_data_' + mode + '_' + identity_string

    root_path = 'E:/alldata'

    radar_point_data_path = os.path.join(root_path, f_dir, 'f_data_points.p')
    radar_voxel_data_path = os.path.join(root_path, f_dir, 'f_data_voxel.p')

    videoData_path = os.path.join(root_path, v_dir, 'cam2')
    mergedImg_path = os.path.join(root_path, identity_string)
    out_path = os.path.join('E:/alldataset', mode + '_' + identity_string)

    return radar_point_data_path, radar_voxel_data_path, videoData_path, mergedImg_path, out_path, identity_string

def generate_train_val_ids(test_ratio, dataset_path='D:/indexPen/dataset', sample_num=None):
    data_ids = os.listdir(dataset_path)

    data_ids = list(map(lambda x: os.path.splitext(x)[0], data_ids))

    # use pre-set random for reproducibility
    random.seed(12)
    random.shuffle(data_ids)

    if sample_num is not None:
        data_ids = data_ids[:sample_num]

    num_data = len(data_ids)
    line = int((1-test_ratio) * num_data)
    train_ids = data_ids[:line]
    test_ids = data_ids[line:]

    data_dict = {'train': [], 'validation': []}

    for train_sample in train_ids:
        data_dict['train'].append(train_sample)

    for test_sample in test_ids:
        data_dict['validation'].append((test_sample))

    return data_dict


def radar_data_grapher_volumned_track(paths, isPlot=False, isCluster=False, augmentation=(),
                                      seeds=np.random.normal(0, 0.02, 5000), timesteps=10, dataset_path = 'F:/thumouse/dataset/'):
    # utility directory to save the pyplots
    radarData_path, videoData_path, mergedImg_path, out_path, identity_string = paths

    radar_3dscatter_path = 'F:/thumouse/figures/utils/'

    radar_data = list(pickle.load(open(radarData_path, 'rb')).items())
    radar_data.sort(key=lambda x: x[0])  # sort by timestamp
    videoData_list = os.listdir(videoData_path)
    videoData_timestamps = list(map(lambda x: float(x.strip('.jpg')), videoData_list))

    style.use('fivethirtyeight')
    white_color = 'rgb(255, 255, 255)'

    DBSCAN_esp = 0.2
    DBSCAN_minSamples = 3

    # input data for the classifier that has the shape n*4*100, n being the number of samples

    fnt = ImageFont.truetype("arial.ttf", 16)

    # removed and recreate the merged image folder
    if isPlot:
        if os.path.isdir(mergedImg_path):
            shutil.rmtree(mergedImg_path)
        os.mkdir(mergedImg_path)

    volume_shape = (25, 25, 25)

    circular_vol_buffer = []

    interval_sec = 5
    sample_per_sec = 20
    sample_per_interval = interval_sec * sample_per_sec

    aug_string = ''
    if augmentation:
        print('Use augmentation: ' + str(augmentation))
        for aug in augmentation:
            aug_string += '_' + aug
    else:
        print('No augmentation applied')

    for i, radarFrame in enumerate(radar_data):

        # retrieve the data
        timestamp, fData = radarFrame

        if isPlot:

            closest_video_timestamp = min(videoData_timestamps,
                                          key=lambda x: abs(x - timestamp))
            closest_video_path = os.path.join(videoData_path, str(closest_video_timestamp) + '.jpg')
            closest_video_img = Image.open(closest_video_path)

            # plot the radar scatter
            ax1 = plt.subplot(2, 2, 1, projection='3d')
            ax1.set_xlim((-0.3, 0.3))
            ax1.set_ylim((-0.3, 0.3))
            ax1.set_zlim((-0.3, 0.3))
            ax1.set_xlabel('X', fontsize=10)
            ax1.set_ylabel('Y', fontsize=10)
            ax1.set_zlabel('Z', fontsize=10)
            ax1.set_title('Detected Points', fontsize=10)
            # plot the detected points
            ax1.scatter(fData['x'], fData['y'], fData['z'], c=fData['doppler'], marker='o')

        data = np.asarray([fData['x'], fData['y'], fData['z'], fData['doppler']]).transpose()
        # Do DBSCAN cluster ###########################################
        # map the points to their doppler value, this is for retrieving the doppler value after clustering
        if isCluster:
            doppler_dict = {}
            for point in data:
                doppler_dict[tuple(point[:3])] = point[3:]
            # get rid of the doppler for clustering TODO should we consider the doppler in clustering?
            data = data[:, :3]

            db = DBSCAN(eps=DBSCAN_esp, min_samples=DBSCAN_minSamples).fit(data)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            if isPlot:
                ax2 = plt.subplot(2, 2, 2, projection='3d')
                ax2.set_xlim((-0.3, 0.3))
                ax2.set_ylim((-0.3, 0.3))
                ax2.set_zlim((-0.3, 0.3))
                ax2.set_xlabel('X', fontsize=10)
                ax2.set_ylabel('Y', fontsize=10)
                ax2.set_zlabel('Z', fontsize=10)
                ax2.set_title('Clustered Points', fontsize=10)

            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]

            clusters = []

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                class_member_mask = (labels == k)
                xyz = data[class_member_mask & core_samples_mask]
                if xyz.any():  # in case there are none objects
                    clusters.append(xyz)  # append this cluster data to the cluster list
                # each cluster is a 3 * n matrix
                xyz = data[class_member_mask & ~core_samples_mask]
                if isPlot:
                    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=12,
                                marker='X')  # plot the noise

            # find the center for each cluster
            clusters_centers = list(
                map(lambda xyz: np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]), clusters))
            clusters.sort(key=lambda xyz: distance.euclidean((0.0, 0.0, 0.0), np.array(
                [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])])))

            # plot the clusters
            for xyz, col in zip(clusters, colors):
                if isPlot:
                    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=28,
                                marker='o')  # plot the cluster points

            #############################
            # clear the hand cluster
            hand_cluster = []
            bbox = (0.2, 0.2, 0.2)

            if len(clusters) > 0:
                hand_cluster = clusters[0]
                point_num = hand_cluster.shape[0]

                # if the cluster is outside the 20*20*20 cm bounding box
                distance_from_center = distance.euclidean((0.0, 0.0, 0.0), np.array(
                    [np.mean(hand_cluster[:, 0]), np.mean(hand_cluster[:, 1]), np.mean(hand_cluster[:, 2])]))

                if distance_from_center > distance.euclidean((0.0, 0.0, 0.0),
                                                             bbox):  # if the core of the cluster is too far away from the center
                    hand_cluster = np.zeros((hand_cluster.shape[0], hand_cluster.shape[1] + 1))
                else:
                    doppler_array = np.zeros((point_num, 1))
                    for j in range(point_num):
                        doppler_array[j:, ] = doppler_dict[tuple(hand_cluster[j, :3])]
                    # append back the doppler
                    hand_cluster = np.append(hand_cluster, doppler_array, 1)
        else:
            hand_cluster = data

        hand_cluster = np.array(hand_cluster)

        # apply augmentation to hand cluster #############################
        if hand_cluster.size != 0:
            # apply augmentations
            if 'trans' in augmentation:
                for p in np.nditer(hand_cluster[:, :3], op_flags=['readwrite']):
                    p[...] = p + random.choice(seeds)
            if 'rot' in augmentation:
                hand_cluster[:, :3] = rotateX(hand_cluster[:, :3], 720 * random.choice(seeds))
                hand_cluster[:, :3] = rotateY(hand_cluster[:, :3], 720 * random.choice(seeds))
                hand_cluster[:, :3] = rotateZ(hand_cluster[:, :3], 720 * random.choice(seeds))
            if 'scale' in augmentation:
                s = 1 + random.choice(seeds)
                hand_cluster[:, :3] = scale(hand_cluster[:, :3], x=s, y=s, z=s)

            if isPlot:
                ax3 = plt.subplot(2, 2, 3, projection='3d')
                ax3.set_xlim((-0.3, 0.3))
                ax3.set_ylim((-0.3, 0.3))
                ax3.set_zlim((-0.3, 0.3))
                ax3.set_xlabel('X', fontsize=10)
                ax3.set_ylabel('Y', fontsize=10)
                ax3.set_zlabel('Z', fontsize=10)
                ax3.set_title('Hand Cluster', fontsize=10)

                ax3.scatter(hand_cluster[:, 0], hand_cluster[:, 1], hand_cluster[:, 2], 'o', c=hand_cluster[:, 3], s=28,
                            marker='o')

        # create 3D feature space #############################
        frame_3D_volume = snapPointsToVolume(hand_cluster, volume_shape, isClipping=('clipping' in augmentation))

        print('Processing ' + str(i + 1) + ' of ' + str(len(radar_data)) + ' Circular buffer size: ' + str(len(circular_vol_buffer)))

        if timesteps == 1:
            this_path = os.path.join(dataset_path, str(timestamp.as_integer_ratio()[0]) + '_' + str(
                timestamp.as_integer_ratio()[1]) + aug_string)
            np.save(this_path, frame_3D_volume)
        else:
            circular_vol_buffer.append(np.expand_dims(frame_3D_volume, axis=0))

            if len(circular_vol_buffer) == timesteps:
                # save this sequence
                print('saving npy...', end='')
                this_path = os.path.join(dataset_path, str(timestamp.as_integer_ratio()[0]) + '_' + str(timestamp.as_integer_ratio()[1]) + aug_string)
                if os.path.exists(this_path):
                    raise Exception('File ' + this_path + ' already exists. THIS SHOULD NEVER HAPPEN!')
                np.save(this_path, circular_vol_buffer)
                print('saved to ' + this_path)
                circular_vol_buffer = circular_vol_buffer[1:]
            elif len(circular_vol_buffer) > timesteps:
                raise Exception('Circular Buffer Overflows. THIS SHOULD NEVER HAPPEN!')

        # Plot the hand cluster #########################################

        #################################################################
        # Combine the three images
        if isPlot:
            plt.savefig(os.path.join(radar_3dscatter_path, str(timestamp) + '.jpg'))
            radar_3dscatter_img = Image.open(os.path.join(radar_3dscatter_path, str(timestamp) + '.jpg'))

            images = [closest_video_img, radar_3dscatter_img]  # add image here to arrange them horizontally
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]

            timestamp_difference = abs(float(timestamp) - float(closest_video_timestamp))
            draw = ImageDraw.Draw(new_im)

            # draw the timestamp difference on the image
            (x, y) = (20, 10)
            message = "Timestamp Difference, abs(rt-vt): " + str(timestamp_difference)
            draw.text((x, y), message, fill=white_color, font=fnt)
            # draw the timestamp
            (x, y) = (20, 30)
            message = "Timestamp: " + str(timestamp)
            draw.text((x, y), message, fill=white_color, font=fnt)

            # draw the number of points
            (x, y) = (20, 60)
            message = "Number of detected points: " + str(xyz.shape[0])
            draw.text((x, y), message, fill=white_color, font=fnt)

            # draw the number of clusters and number of noise point on the clutter plot
            if isCluster:
                (x, y) = (20, 80)
                message = "Number of clusters: " + str(n_clusters_)
                draw.text((x, y), message, fill=white_color, font=fnt)
                (x, y) = (20, 100)
                message = "Number of outliers: " + str(n_noise_)
                draw.text((x, y), message, fill=white_color, font=fnt)

            # save the combined image
            new_im.save(
                os.path.join(mergedImg_path, str(timestamp) + '_' + str(timestamp.as_integer_ratio()[0]) +
                             '_' + str(timestamp.as_integer_ratio()[1]) + '.jpg'))
            plt.close('all')
