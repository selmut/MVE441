import numpy as np
import pandas as pd
from Clustering.DBSCAN import DensityBasedClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score
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
    #std_data=StandardScaler().fit_transform(df)
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

max_dim=30
n_min = 8
dimCluster=np.empty(max_dim)
dimNMI=np.empty(max_dim)
dimSil=np.empty(max_dim)
dimFM=np.empty(max_dim)


for dim in range(1,max_dim):
    pca_data=reduce_dim(data,dim)
    maxNeighbourDist=np.zeros(82)

    for i in range(82):
        distances=np.sqrt(np.sum((pca_data-pca_data[i,:])**2,axis=1))
        idx = np.argpartition(distances, n_min)
        maxNeighbourDist[i]=distances[idx[n_min]]


    epsilons=np.arange(np.round(np.min(maxNeighbourDist),1),np.round(np.max(maxNeighbourDist),1),step=0.1)

    n=len(epsilons)

    nmi=np.zeros(n)
    fm=np.zeros(n)
    silhouette=np.zeros(n)
    clusters=np.zeros(n)

    for i,eps in enumerate(epsilons):
        dbscan=DensityBasedClustering(eps,n_min)
        result=dbscan.fit_data(pca_data)

        nmi[i]=normalized_mutual_info_score(labels,result)
        fm[i]=fowlkes_mallows_score(labels, result)
        nr_clusters = np.max(result) + 1
        if nr_clusters>1:
            silhouette[i] = silhouette_score(pca_data,result)
        clusters[i]= nr_clusters

    bestEpsidx=np.argmax(nmi)
    dimCluster[dim]=clusters[bestEpsidx]
    dimNMI[dim]=nmi[bestEpsidx]
    dimSil[dim]=silhouette[bestEpsidx]
    dimFM[dim]=fm[bestEpsidx]

    if dim==5:
        dbscan = DensityBasedClustering(epsilons[bestEpsidx], n_min)
        result = dbscan.fit_data(pca_data)
        print("unclassified:", np.sum(result == -1))
        accuracy(result, labels)
        print("nmi score:", dimNMI[5])
        print("silhouette score:", dimSil[5])
        print("fowlkes malloes score:", dimFM[5])

        plt.plot(np.arange(82),np.sort(maxNeighbourDist))
        plt.ylabel('epsilon')
        plt.show()

        plt.plot(epsilons,nmi, label='NMI-score')
        plt.plot(epsilons,fm, label='FM-score')
        plt.plot(epsilons,silhouette,label='Silhouette-score')
        plt.xlabel('epsilon')
        plt.legend()
        plt.show()

        plt.scatter(epsilons,clusters)
        plt.xlabel('epsilon')
        plt.ylabel('nr of clusters')
        plt.show()

plt.plot(np.arange(1,max_dim), dimNMI[1:],label='NMI-score')
plt.plot(np.arange(1,max_dim), dimSil[1:],label='Silhouette-score')
plt.plot(np.arange(1,max_dim), dimFM[1:],label='FM-score')
plt.xlabel('PCA-dim')
plt.legend()
plt.show()

plt.scatter(np.arange(1,max_dim), dimCluster[1:])
plt.xlabel('PCA-dim')
plt.ylabel('nr of clusters')
plt.show()


#ax = plt.axes(projection='3d')
#ax.scatter3D(pca_data[:,0], pca_data[:,1], pca_data[:,2], c=labels)
#plt.show()


#plt.scatter(pca_data[:,0], pca_data[:,1], c=labels)
#plt.show()