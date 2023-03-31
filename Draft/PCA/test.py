import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from PCAclass import SPDRePCA
import seaborn as sns


data = np.load("PCA/cleandata.npy", allow_pickle=True)
data_T = data.T
wavelengths = np.arange(360, 831)
mu = np.array([415, 445, 480, 515, 555, 590, 630, 680])
sigma = np.array([11.0403, 12.7388, 15.2866, 16.5605, 16.5605, 16.9851, 21.2314, 22.0807])

kmeans = KMeans(2,n_init = 10,random_state=0)
labels = kmeans.fit_predict(data_T)

#可视化聚类结果
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=data_T[:, 0], y=data_T[:, 1], hue=labels, palette="tab10", ax=ax)
ax.set_title("K-Means Clustering Result")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.show()


"""cluster_0 = data_T[labels == 0].T
cluster_1 = data_T[labels == 1].T

total = np.concatenate((cluster_0,cluster_1), axis =1)

print(total.shape)


print(cluster_0.shape)


Re1 = SPDRePCA(cluster_0, 8, wavelengths, mu, sigma)
Re1.Reconstructed_spectrum()
#Re2 = SPDRePCA(cluster_1, 8, wavelengths, mu, sigma)
#Re2.Reconstructed_spectrum()
R = SPDRePCA(total, 8, wavelengths, mu, sigma) 
R.Reconstructed_spectrum()"""


#Re1.Plot(3)
#Re1.Evaluate(3)

Re1.Plot(93)
Re1.Evaluate(93)
R.Plot(93)
R.Evaluate(93)




"""Re2.Reconstructed_spectrum()
Re2.Plot(3)
Re2.Evaluate(3)

Re2.Plot(93)
Re2.Evaluate(93)"""
#print(labels[0],labels[1] ,labels[100],)

"""target_sample1 = data[:,1].T  # 假设目标样本的特征值是这个列表
indices = np.where(np.all(cluster_1 == target_sample1, axis=0))[0]
print(cluster_0.shape)
print(indices)"""