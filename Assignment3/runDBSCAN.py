import numpy as np
import pandas as pd
from Classifiers.kmeans import kMeans
from Classifiers.DBSCAN import DensityBasedClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from loocv import LOOCV
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


df = pd.read_csv('data/Cancerdata.txt', sep="\t")
new_keys = {'lab': 'label'}
new_keys.update(dict(zip([f'Unnamed: {i}' for i in range(3000)], [f'Gene {i}' for i in range(3000)])))

df = df.rename(columns=new_keys)

data = df.loc[:, 'Gene 1':'Gene 2999']
labels = df['label']

def reduce_dim(df,n):
    std_data=StandardScaler().fit_transform(df)
    pca=PCA(n_components=n,svd_solver='full')
    return pca.fit_transform(df)


def accuracy(clusteredData,labels):
    totalAccuracy = 0
    nr_clusters=np.max(clusteredData)+1
    for n in range(nr_clusters):
        cluster=np.where(clusteredData==n)[0]
        clusterLabs=labels.iloc[cluster]
        majorityLabel=clusterLabs.value_counts().idxmax()
        acc=np.sum(clusterLabs==majorityLabel)/len(clusterLabs)
        totalAccuracy=totalAccuracy+acc
        print("cluster:",n, "label:",majorityLabel, "accuracy:",acc)

    return totalAccuracy/nr_clusters

pca_data=reduce_dim(data,5)

#8, 12.9

n_min=6
d=np.zeros(82)

for i in range(82):
    distances=np.sqrt(np.sum((pca_data-pca_data[i,:])**2,axis=1))
    idx = np.argpartition(distances, n_min)
    d[i]=distances[idx[n_min]]

plt.plot(np.arange(82),np.sort(d))
plt.show()

dbscan=DensityBasedClustering(12,n_min)
result=dbscan.fit_data(pca_data)

print(result)
accuracy(result,labels)
