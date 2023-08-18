"""Code implementation by Soheil Abadifard - 22101026"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


class k_means(object):

    # initialize with some values
    def __init__(self, k, data):

        self.number_of_clusters = k
        self.dimensions = 3
        self.centers = np.zeros(shape=(self.number_of_clusters, self.dimensions))

        self.data = data
        self.labels = []
        self.error = 0
        self.clustered_data = []
        # picking some values randomly from image as starting point
        for i in range(0, self.number_of_clusters):
            index = np.random.rand() * self.data.shape[0]
            index = int(index)
            print(index)
            print(self.data[index])
            self.centers[i] = self.data[index]

    # calculating distance
    def euclidean_distance(self, mydata, cluster):
        eu_pow = 0
        eu_distance = 0
        eu_sum = 0
        eu_pow = (mydata - self.centers[cluster])

        eu_pow = eu_pow ** 2
        eu_pow = np.array(eu_pow)
        eu_sum = eu_pow.sum(axis=1)
        eu_distance = np.sqrt(eu_sum)
        return eu_distance

    # clustering the data
    def cluster(self):
        distances = []
        for i in range(0, self.number_of_clusters):
            distances.append(self.euclidean_distance(self.data, i))
        distances = np.array(distances)

        self.labels = np.argmin(distances, axis=0)
        self.labels = self.labels.reshape(self.labels.shape[0], 1)
        labeled_data = np.append(self.data, self.labels, axis=1)
        # print(self.labels[0:100])
        clustred_data_2 = []
        for i in range(0, self.number_of_clusters):
            tmp = np.where(labeled_data[:, 3] == i)

            tmp = np.array(tmp)
            tmp2 = self.data[tmp, :]
            clustred_data_2.append(tmp2)
            clustred_data_2[i] = np.array(clustred_data_2[i])
            clustred_data_2[i] = clustred_data_2[i].reshape(clustred_data_2[i].shape[1], clustred_data_2[i].shape[2])
        clustred_data_2 = np.array(clustred_data_2)
        clustred_data = np.array(clustred_data_2)
        self.clustered_data = clustred_data

        i = 0
        for c in clustred_data:
            v = c.sum(axis=0)
            self.centers[i] = v / (c.shape[0] + 0.00000001)
            i = i + 1

    # here as i mentioned in the report I use 10 as my number of iteration and threshold
    def find_best_cluster(self):
        threshold = 19
        for i in range(0, threshold):
            self.cluster()

    # calculating error
    def sum_of_squarred_error(self, clustered_data):
        for i in range(0, self.number_of_clusters):
            mean_data = clustered_data[i].mean()
            e = np.linalg.norm(clustered_data[i] - mean_data)
            e = e ** 2
            e = e.sum()
            self.error += e


img = cv2.imread("sample.jpg")

x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = x.astype(np.float32)
x = x.reshape((-1, 3))

start = time.time()
clustering = k_means(6, x)
clustering.find_best_cluster()

centers = clustering.centers
label = clustering.labels

labeled_data = np.concatenate((x, clustering.labels), axis=1)
end = time.time()
print(" time :", end - start)

centers = np.uint8(centers)
segmented = centers[label]
segmented1 = segmented.reshape(img.shape)

print(centers)
clustred_data_2 = []
for i in range(0, clustering.number_of_clusters):
    tmp = np.where(labeled_data[:, 3] == i)

    tmp = np.array(tmp)
    tmp2 = x[tmp, :]
    clustred_data_2.append(tmp2)
    clustred_data_2[i] = np.array(clustred_data_2[i])
    clustred_data_2[i] = clustred_data_2[i].reshape(clustred_data_2[i].shape[1], clustred_data_2[i].shape[2])
clustred_data_2 = np.array(clustred_data_2)
clustred_data = np.array(clustred_data_2)

clustering.sum_of_squarred_error(clustred_data)

print("error : ", clustering.error)

plt.imshow(segmented1)

plt.imsave("k 2- fast .png", segmented1)
