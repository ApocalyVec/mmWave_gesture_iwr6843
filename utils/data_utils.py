import pickle

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import unique_labels

from utils.path_utils import snapPointsToVolume


volume_shape = [25, 25, 25]


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

DBSCAN_esp = 0.2
DBSCAN_minSamples = 3
def produce_voxel(points, isCluster=True, isClipping=False):
    """

    :param frame: np array with input shape (n, 4)
    :return voxel
    """
    if len(points) == 0:  # if there's no detected points
        return np.zeros(tuple([1] + volume_shape))

    if isCluster:
        doppler_dict = {}
        for point in points:
            doppler_dict[tuple(point[:3])] = point[3:]
        # get rid of the doppler for clustering TODO should we consider the doppler in clustering?
        points = points[:, :3]

        db = DBSCAN(eps=DBSCAN_esp, min_samples=DBSCAN_minSamples).fit(points)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        unique_labels = set(labels)
        clusters = []
        for k in zip(unique_labels):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xyz = points[class_member_mask & core_samples_mask]
            if xyz.any():  # in case there are none objects
                clusters.append(xyz)  # append this cluster data to the cluster list
            # each cluster is a 3 * n matrix
            xyz = points[class_member_mask & ~core_samples_mask]

        # find the center for each cluster
        # clusters_centers = list(
        #     map(lambda xyz: np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]), clusters))
        clusters.sort(key=lambda xyz: distance.euclidean((0.0, 0.0, 0.0), np.array(
            [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])])))

        #############################
        hand_cluster = []
        if len(clusters) > 0:
            hand_cluster = clusters[0]
            point_num = hand_cluster.shape[0]

            # if the cluster is outside the 20*20*20 cm bounding box
            distance_from_center = distance.euclidean((0.0, 0.0, 0.0), np.array(
                [np.mean(hand_cluster[:, 0]), np.mean(hand_cluster[:, 1]), np.mean(hand_cluster[:, 2])]))

            # if distance_from_center > distance.euclidean((0.0, 0.0, 0.0),
            #                                              bbox):  # if the core of the cluster is too far away from the center
            #     hand_cluster = np.zeros((hand_cluster.shape[0], hand_cluster.shape[1] + 1))
            doppler_array = np.zeros((point_num, 1))
            for j in range(point_num):
                doppler_array[j:, ] = doppler_dict[tuple(hand_cluster[j, :3])]
            # append back the doppler
            hand_cluster = np.append(hand_cluster, doppler_array, 1)
    else:
        hand_cluster = points

    hand_cluster = np.array(hand_cluster)
    frame_3D_volume = snapPointsToVolume(hand_cluster, volume_shape, isClipping=isClipping)

    return np.expand_dims(frame_3D_volume,axis=0)

# frameArray = np.load('F:/test_frameArray.npy')
# start = time.time()
# result = preprocess_frame(frameArray[2])
# end = time.time()
# print('Preprocessing frame took ' + str(end-start))