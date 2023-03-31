import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from PCAclass import SPDRePCA


class SPDReCluster:
    def __init__(self, spd):
        self.data = spd #(471,1494)
        self.data_T = spd.T
        self.best_n_clusters = None
        self.labels_ = None
        self.Group = None
        self.A_hat = None
        self.mu = np.array([415, 445, 480, 515, 555, 590, 630, 680])
        self.sigma = np.array([11.0403, 12.7388, 15.2866, 16.5605, 16.5605, 16.9851, 21.2314, 22.0807])
        self.wavelengths = np.arange(360, 831)

    
    def clusternumber(self):
        # 为聚类结果保存轮廓系数
        silhouette_scores = []
        # 选择聚类数从2到10
        for n_clusters in range(2, 11):
            # 计算KMeans聚类
            kmeans = KMeans(n_clusters=n_clusters, n_init=10 ,random_state=0)
            cluster_labels = kmeans.fit_predict(self.data_T)
            # 计算轮廓系数
            silhouette_avg = silhouette_score(self.data_T, cluster_labels)
            # 将轮廓系数保存到列表中
            silhouette_scores.append(silhouette_avg)
        # 找到具有最大轮廓系数的聚类数
        self.best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    def fit(self):
        self.model = KMeans(self.best_n_clusters, n_init=10)
        self.model.fit(self.data_T)
        self.labels_ = self.model.labels_
        

    def divideData(self):
        self.Group = {}  # 定义一个空字典
        # 根据需求动态命名变量并添加到字典中
        for i in range(self.best_n_clusters):
            variable_name = f"Cluster_{i}"  # 构造变量名
            n = 471
            cluster_data = np.empty([0,n])
            for k in range(len(self.labels_)):
                if self.labels_[k] == i:
                    cluster_data = np.vstack([cluster_data, self.data_T[k]])
                
            print(k)
            self.Group[variable_name] = cluster_data.T  # 添加到字典中
        
        #融合两个cluster, 重构SPDdata
        clusted_spd_Datalist =[]
        for k in self.Group.keys():
            clusted_spd_Datalist.append(self.Group[k])
        self.combined_spd = np.concatenate(clusted_spd_Datalist, axis=1)


    
    def compute(self):
        self.A_hat_dict = {}
        self.REconstruct = {}
        if self.Group is None:
            print("No data found. Please run 'divideData()' function first.")
            return
        key_name = self.Group.keys()
        for group_name in key_name:
            cluster_temp_data = self.Group[group_name]
            spectrum = SPDRePCA(cluster_temp_data, 8, self.wavelengths, self.mu, self.sigma)
            spectrum.Reconstructed_spectrum()
            a_hat = spectrum.a_hat
            self.A_hat_dict[group_name] = a_hat
            self.REconstruct[group_name] = spectrum.reconstructed_spectrum
        
        #merge
        clusted_REspd_Datalist =[]
        for k in self.REconstruct.keys():
            clusted_REspd_Datalist.append(self.REconstruct[k])
        self.combined_REspd = np.concatenate(clusted_REspd_Datalist, axis=1)

         
    def Plot(self, Light):
        # 创建一个包含三个子图的图像
        fig, axs = plt.subplots(3, 1)
        s = self.combined_spd[:,Light]
        r = self.combined_REspd[:,Light]

        # 绘制真实光谱图
        axs[0].plot(self.wavelengths, s, label='True Spectrum')
        axs[0].set_xlabel('Wavelength (nm)')
        axs[0].set_ylabel('Intensity')
        axs[0].set_title('True Spectrum vs Reconstructed Spectrum')
        axs[0].legend()

        # 绘制重建光谱图
        axs[1].plot(self.wavelengths, r, label='Reconstructed Spectrum')
        axs[1].set_xlabel('Wavelength (nm)')
        axs[1].set_ylabel('Intensity')
        axs[1].legend()

        # 绘制重建光谱和真实光谱叠加的图
        axs[2].plot(self.wavelengths, s, label='True Spectrum')
        axs[2].plot(self.wavelengths, r, label='Reconstructed Spectrum')
        axs[2].set_xlabel('Wavelength (nm)')
        axs[2].set_ylabel('Intensity')
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    def Evaluate(self, Light):
        s = self.combined_spd[:,Light]
        s_re = self.combined_REspd[:,Light]

        MAE = np.mean(np.absolute(s - s_re))
        RMSE = np.sqrt(np.mean(np.square(s- s_re)))
        RRMSE = RMSE / np.mean(s_re)

        #GFC
        num = abs(np.sum(s * s_re))
        denom_s = np.sqrt(np.sum(s ** 2) + 1e-9)
        denom_s_re = np.sqrt(np.sum(s_re ** 2) + 1e-9)
        GFC =num / (denom_s * denom_s_re)

        print("MAE: ", MAE)
        print("RMSE: ", RMSE)
        print("RRMES: ", RRMSE)
        print("GFC: ", GFC)



    def predict(self, new_data):
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)
        new_data_cluster = self.model.predict(new_data)[0]
        return new_data_cluster
    

if __name__ == "__main__":

    spd = np.load("PCA/cleandata.npy", allow_pickle=True)
    r = SPDReCluster(spd)
    r.clusternumber()
    r.fit()
    r.divideData()
    r.compute()
     
    """list1 = [1,50,99, 300,450]
    list2 = [700, 900, 1200,1329,1400]

    for i in list1:
        print(f"number{i}:")
        r.Evaluate(i)
        r.Plot(i)"""
    
    

    