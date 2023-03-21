import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from PCAclass import SPDReconstruction


spd = np.load("PCA/cleandata.npy", allow_pickle=True)

spd_T = spd.T


wavelengths = np.arange(360, 831)
mu = np.array([415, 445, 480, 515, 555, 590, 630, 680])
sigma = np.array([11.0403, 12.7388, 15.2866, 16.5605, 16.5605, 16.9851, 21.2314, 22.0807])


n_clusters = 2  # 通过肘部法则或轮廓系数法确定的最佳聚类数

kmeans = KMeans(n_clusters=n_clusters)  # 初始化KMeans模型
kmeans.fit(spd_T)  # 对数据集进行聚类
cluster_labels = kmeans.labels_  # 获取每个样本所属簇的标签 (1494,)

n = 471
cluster_data1 = np.empty([0,n])
cluster_data2 = np.empty([0,n])

for i in cluster_labels:
    if i == 0:
        cluster_data1 = np.vstack([cluster_data1, spd_T[i]])
    if i == 1:
        cluster_data2 = np.vstack([cluster_data2, spd_T[i]])

spd1 = cluster_data1.T
spd2 = cluster_data2.T

r = SPDReconstruction(spd1, 8, wavelengths, mu, sigma)
r.Reconstructed_spectrum()
r.Plot(103)
r.Evaluate(103)