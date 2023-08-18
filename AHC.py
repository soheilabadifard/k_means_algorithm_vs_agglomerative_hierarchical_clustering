"""Code implementation by Soheil Abadifard - 22101026"""

import time
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np


class agglomerative(object):

    def __init__(self, k, data):

        self.number_of_clusters = k
        self.dimensions = 3
        self.data = data
        self.labels = np.zeros(shape=(self.data.shape[0], 1))
        self.children = []
        self.node = np.zeros(shape=(self.data.shape[0], 1))
        self.number_of_leaves = []
        self.final_nodes = []
        self.node_data = []
        self.error = 0

    # calculating euclidean distance
    def euclidean_distance(self, mydata):
        eu_pow = 0
        eu_distance = 0
        eu_sum = 0
        eu_pow = self.data - mydata
        eu_pow = eu_pow ** 2
        eu_pow = np.array(eu_pow)
        eu_sum = eu_pow.sum(axis=1)
        eu_distance = np.sqrt(eu_sum)
        return eu_distance

    def euclidean_distance2(self, mydata, center):

        eu_pow = 0
        eu_distance = 0
        eu_sum = 0
        eu_pow = (mydata - center)
        eu_pow = eu_pow ** 2
        eu_pow = np.array(eu_pow)
        eu_sum = eu_pow.sum(axis=1)
        eu_distance = np.sqrt(eu_sum)
        return eu_distance

    # doing the hiearchial clustering
    def cluster(self):
        counter = 0
        distance_matrix = [self.euclidean_distance(i) for i in self.data]
        distance_matrix = np.array(distance_matrix)
        np.fill_diagonal(distance_matrix, float('Inf'))

        distance_matrix = np.array(distance_matrix)

        for i in range(0, self.data.shape[0] - 1):
            x = np.min(distance_matrix)

            min_location = np.where(distance_matrix == x)
            min_location = np.array(min_location)
            tmp_min_loc = []
            if min_location.shape[1] > 2:
                # min_location = np.delete(min_location,[1,2,3],1)
                min_location = min_location[:, 0]
                min_location = min_location.reshape(2, 1)
                rev_1 = min_location[1, 0]
                rev_2 = min_location[0, 0]
                tmp_min_loc.append(rev_1)
                tmp_min_loc.append(rev_2)
                tmp_min_loc = np.array(tmp_min_loc)
                tmp_min_loc = tmp_min_loc.reshape(2, 1)
                min_location = np.concatenate((min_location, tmp_min_loc), axis=1)

            if self.node[min_location[0, 0]] == 0 and self.node[min_location[0, 1]] == 0:
                self.children.append(np.array([min_location[0, 0], min_location[0, 1]]))
                self.number_of_leaves.append(2)
                self.node_data.append(np.array([min_location[0, 0], min_location[0, 1]]))
                # print(self.children)
            elif self.node[min_location[0, 0]] == 0 and self.node[min_location[0, 1]] != 0:

                a = int(min_location[0, 0])
                b = int(self.node[min_location[0, 1]])

                a_ = min(a, b)
                b_ = max(a, b)
                child = []
                child.append(a_)
                child.append(b_)
                child = np.array(child)
                self.children.append(child)
                self.number_of_leaves.append((1 + self.number_of_leaves[b - self.data.shape[0]]))

                first = min_location[0, 0]
                second = self.node_data[b_ - self.data.shape[0]]
                this_node = np.hstack((first, second))
                self.node_data.append(this_node)

            elif self.node[min_location[0, 0]] != 0 and self.node[min_location[0, 1]] == 0:

                a = int(self.node[min_location[0, 0]])
                b = int(min_location[0, 1])
                a_ = min(a, b)
                b_ = max(a, b)
                child = []
                child.append(a_)
                child.append(b_)
                child = np.array(child)
                self.children.append(child)

                self.number_of_leaves.append((1 + self.number_of_leaves[a - self.data.shape[0]]))

                first = self.node_data[b_ - self.data.shape[0]]
                second = min_location[0, 1]
                this_node = np.hstack((first, second))
                self.node_data.append(this_node)

            else:
                a = int(self.node[min_location[0, 0]])
                b = int(self.node[min_location[0, 1]])
                a_ = min(a, b)
                b_ = max(a, b)
                child = []
                child.append(a_)
                child.append(b_)
                child = np.array(child)
                self.children.append(child)
                self.number_of_leaves.append(
                    (self.number_of_leaves[b - self.data.shape[0]] + self.number_of_leaves[a - self.data.shape[0]]))

                first = self.node_data[a_ - self.data.shape[0]]
                second = self.node_data[b_ - self.data.shape[0]]
                this_node = np.hstack((first, second))
                # this_node = this_node.reshape(1,this_node.shape[1])

                self.node_data.append(this_node)

            self.node[min_location[0, 0]] = self.data.shape[0] + counter
            self.node[min_location[0, 1]] = self.data.shape[0] + counter

            two_rows = []
            two_rows = (distance_matrix[min_location[0, 0:2]])

            # average Linkage

            two_rows = np.sum(two_rows, axis=0)
            two_rows = two_rows / 2

            two_rows = np.array(two_rows)

            distance_matrix[min_location[0, 1]] = two_rows
            distance_matrix[:, min_location[0, 1]] = two_rows

            distance_matrix[min_location[0, 0]] = float('Inf')
            distance_matrix[:, min_location[0, 0]] = float('Inf')
            np.fill_diagonal(distance_matrix, float('Inf'))

            counter = counter + 1

        self.children = np.array(self.children)

        root = self.children[-1]
        root = np.array(root)
        self.final_nodes = np.array(self.final_nodes)
        limit = self.number_of_clusters

        if self.number_of_clusters == 1:
            self.final_nodes = np.hstack([self.final_nodes, [self.data.shape[0] + self.children.shape[0]]])
        elif self.number_of_clusters == 2:
            self.final_nodes = np.hstack([self.final_nodes, root[0]])
            self.final_nodes = np.hstack([self.final_nodes, root[1]])
        if self.number_of_clusters == self.data.shape[0]:
            root = [0] * self.data.shape[0]

        else:
            while root.shape[0] < limit:
                node = []
                for i in root:
                    node = np.max(root)
                i = np.where(root == np.max(node))
                i = int(i[0])
                root[i] = self.children[(np.max(node) - self.data.shape[0])][0]
                root = np.insert(root, i + 1, self.children[(np.max(node) - self.data.shape[0])][1])

        root = np.array(root)
        self.final_nodes = root
        self.node_data = np.array(self.node_data)

        self.node_data = self.node_data.reshape(self.node_data.shape[0], 1)

        label = int(0)

        for node in self.final_nodes:

            if self.number_of_clusters == self.data.shape[0]:
                self.labels = np.arange(0, self.data.shape[0])
                break
            elif node > self.data.shape[0] - 1:
                n = node - self.data.shape[0]
                cluster = self.node_data[n]
                self.labels[cluster[0]] = label
            else:
                cluster = node
                cluster = np.array(cluster)
                self.labels[cluster] = label

            label += 1

        self.labels = np.array(self.labels)
        self.labels = self.labels.astype(int)
        self.labels = self.labels.reshape(1, self.labels.shape[0])

    def sum_of_squarred_error(self, clustered_data):
        for i in range(0, self.number_of_clusters):
            mean_data = clustered_data[i].mean()
            e = np.linalg.norm(clustered_data[i] - mean_data)
            e = e ** 2
            e = e.sum()
            self.error += e
        return self.error


img = cv2.imread("sample.jpg")
x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = x.astype(np.float32)
x = x.reshape((-1, 3))

labels = pd.read_csv("labels - 800.csv",header=None)
labels = labels.to_numpy()

centers = pd.read_csv("centers - 800.csv", header=None)
centers = centers.to_numpy()

labels = labels.astype(int)
centers = centers.astype(int)

c = np.array(centers)

start = time.time()

agglo = agglomerative(20, c)
agglo.cluster()

agglo_labels = np.array(agglo.labels)

labeled_data = np.append(c, agglo_labels.T, axis=1)

clustred_data_2 = []
for i in range(0, agglo.number_of_clusters):
    tmp = np.where(labeled_data[:, 3] == i)

    tmp = np.array(tmp)
    tmp2 = c[tmp, :]
    clustred_data_2.append(tmp2)
    clustred_data_2[i] = np.array(clustred_data_2[i])
    clustred_data_2[i] = clustred_data_2[i].reshape(clustred_data_2[i].shape[1], clustred_data_2[i].shape[2])
clustred_data_2 = np.array(clustred_data_2)
clustred_data = np.array(clustred_data_2)

agglo_centers = np.zeros(shape=(agglo.number_of_clusters, agglo.dimensions))
i = 0
for m in clustred_data:
    v = m.sum(axis=0)
    agglo_centers[i] = v / (m.shape[0] + 0.00000001)
    i = i + 1
error = agglo.sum_of_squarred_error(clustred_data)

stop = time.time()

print("ERROR :", error)
print("Time :", stop - start)

agglo_centers = np.uint8(agglo_centers)
agglo_labels = agglo_labels.astype(int)
pre_segmented = agglo_centers[agglo_labels]
pre_segmented = np.squeeze(pre_segmented)
# pre_segmented1 = pre_segmented.reshape(img.shape)
# second phase - apply the new centers to original image

segmented_image = pre_segmented[labels]
segmented_image1 = segmented_image.reshape(img.shape)
print(" Centers :", agglo_centers)
plt.imshow(segmented_image1)
plt.imsave("K 40 - agglo.png", segmented_image1)
