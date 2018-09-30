import sys
import os

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math


def read_data(filepath):
	'''Read data points from file specified by filepath
	Args:
		filepath (str): the path to the file to be read

	Returns:
		numpy.ndarray: a numpy ndarray with shape (n, d) where n is the number of data points and d is the dimension of the data points

	'''

	X = []
	with open(filepath, 'r') as f:
		lines = f.readlines()
		for line in lines:
			X.append([float(e) for e in line.strip().split(',')])
	return np.array(X)


def distance(x, y):
    dis = abs(x-y)
    return round(math.sqrt((dis*dis.T).sum()), 3)


def is_neighborhood(x, y, eps):
    return distance(x, y) <= eps


def range_query(X, point_id, eps):
    neighbors = np.array([X[point_id]])
    for nei_id in range(0,len(X)):

        if (point_id != nei_id):
            if is_neighborhood(X[point_id], X[nei_id], eps):
                neighbors = np.append(neighbors, np.array([X[nei_id]]), axis=0)

    return np.array(neighbors)


def search_neighbors(X, indexes, Neighbors, corepoints, labels, cluster_id, eps, minpts):
    Seeds = Neighbors[1:]
    for seed_id in range(0, len(Seeds)):

        if tuple(Seeds[seed_id]) in labels:
            if labels[tuple(Seeds[seed_id])] == -1:
                labels[tuple(Seeds[seed_id])] = cluster_id
            else:
                continue
        labels[tuple(Seeds[seed_id])] = cluster_id
        nei_neighbors = range_query(X, indexes[tuple(Seeds[seed_id])], eps)
        if len(nei_neighbors) >= minpts:
            # print([indexes[tuple(Seeds[seed_id])]])
            corepoints.extend([indexes[tuple(Seeds[seed_id])]])
            # Neighbors = np.append(Neighbors, nei_neighbors, axis=0)
            labels, corepoints = search_neighbors(X, indexes, nei_neighbors, corepoints, labels, cluster_id, eps, minpts)
    return labels, corepoints



# To be implemented
def dbscan(X, eps, minpts):
    '''dbscan function for clustering
	Args:
		X (numpy.ndarray): a numpy array of points with dimension (n, d) where n is the number of points and d is the dimension of the data points
		eps (float): eps specifies the maximum distance between two samples for them to be considered as in the same neighborhood
		minpts (int): minpts is the number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself.
	
	Returns:
		list: The output is a list of two lists, the first list contains the cluster label of each point, where -1 means that point is a noise point, the second list contains the indexes of the core points from the X array.
	
	Example:
		Input: X = np.array([[-10.1,-20.3], [2.0, 1.5], [4.3, 4.4], [4.3, 4.6], [4.3, 4.5], [2.0, 1.6], [2.0, 1.4]]), eps = 0.1, minpts = 3
		Output: [[-1, 1, 0, 0, 0, 1, 1], [1, 4]]
		The meaning of the output is as follows: the first list from the output tells us: X[0] is a noise point, X[1],X[5],X[6] belong to cluster 1 and X[2],X[3],X[4] belong to cluster 0; the second list tell us X[1] and X[4] are the only two core points

	'''

    # cluster_id = 0
    # labels = {}
    # corepoints = {}
    #
    # for point_id in range(0, len(X)):
    #     if not tuple(X[point_id]) in labels:
    #         expand_success, newlabels= expand_cluster(X,labels,point_id,cluster_id,eps,minpts)
    #         if expand_success:
    #             corepoints[cluster_id] = point_id
    #             cluster_id += 1
    #             labels = newlabels
    #
    # label_list = []
    # for point_id in range(0, len(X)):
    #     label_list.append(labels[tuple(X[point_id])])
    #
    # corepoints_list = []
    # for i in range(0,len(corepoints)):
    #     corepoints_list.append(corepoints[i])
    #
    # return [label_list, corepoints_list]

    cluster_id = 0
    indexes = {}
    corepoints_list = []

    for point_id in range(0, len(X)):
        indexes[tuple(X[point_id])] = point_id

    label = {}

    for point_id in range(0, len(X)):

        if tuple(X[point_id]) in label:
            continue
        Neighbors = range_query(X, point_id, eps)

        #TODO
        if len(Neighbors) < minpts:
            label[tuple(X[point_id])] = -1
            continue

        label[tuple(X[point_id])] = cluster_id
        corepoints_list.append(point_id)

        lablel, corepoints_list = search_neighbors(X, indexes, Neighbors, corepoints_list, label, cluster_id, eps, minpts)
        # Seeds = Neighbors[1:]
        #
        # for seed_id in range(0, len(Seeds)):
        #
        #     if tuple(Seeds[seed_id]) in label:
        #         if label[tuple(Seeds[seed_id])] == -1:
        #             label[tuple(Seeds[seed_id])] = cluster_id
        #         else:
        #             continue
        #     label[tuple(Seeds[seed_id])] = cluster_id
        #     nei_neighbors = range_query(X, seed_id, eps)
        #     if len(nei_neighbors) >= minpts:
        #         Seeds = np.append(Seeds, nei_neighbors, axis = 0)

        cluster_id += 1

    label_list = []
    for point_id in range(0, len(X)):
        label_list.append(label[tuple(X[point_id])])

    # corepoints_list = []
    # for i in range(0,len(corepoints)):
    #     corepoints_list.append(corepoints[i])

    # print(label_list)
    corepoints_list.sort()
    return [label_list, corepoints_list]


def main():
    # X = np.array([[-10.1,-20.3], [2.0, 1.5], [4.3, 4.4], [4.3, 4.6], [4.3, 4.5], [2.0, 1.6], [2.0, 1.4]])
    # print(dbscan(X,  0.1, 3))

    # X = read_data("/Users/lynnjiang/liuGit/DataMining/Assignment 1/DBScan/Data/data_1.txt")
    # label_list, corepoints_list = dbscan(X, 0.3, 10)
    # print(label_list)
    # print(corepoints_list)

    if len(sys.argv) != 4:
        print("Wrong command format, please follwoing the command format below:")
        print("python dbscan-template.py data_filepath eps minpts")
        exit(0)

    X = read_data(sys.argv[1])

    # Compute DBSCAN
    db = dbscan(X, float(sys.argv[2]), int(sys.argv[3]))

    # store output labels returned by your algorithm for automatic marking
    with open('.' + os.sep + 'Output' + os.sep + 'labels.txt', "w") as f:
        for e in db[0]:
            f.write(str(e))
            f.write('\n')

    # store output core sample indexes returned by your algorithm for automatic marking
    with open('.' + os.sep + 'Output' + os.sep + 'core_sample_indexes.txt', "w") as f:
        for e in db[1]:
            f.write(str(e))
            f.write('\n')

    _, dimension = X.shape

    # plot the graph is the data is dimensiont 2
    if dimension == 2:
        core_samples_mask = np.zeros_like(np.array(db[0]), dtype=bool)
        core_samples_mask[db[1]] = True
        labels = np.array(db[0])

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig('.' + os.sep + 'Output' + os.sep + 'cluster-result.png')



main()