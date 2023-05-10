import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings
from matplotlib.ticker import MaxNLocator
import pyreadr
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore")
# Loads the data
TCGAData = pyreadr.read_r('data/TCGAdata.RData')

data = TCGAData['TCGA']
labels = TCGAData['TCGAclassstr']

datapoints = data.shape[0]
features = data.shape[1]

#print(labels.value_counts())



def getImportantFeats(df,n):
    # extracts feature with maximum variance
    var_data = df.to_numpy()
    var_feature = np.var(var_data, axis=0)
    idx = np.argpartition(-var_feature, n)
    orderdidx=np.argsort(-var_feature[idx[:n]])
    return idx[orderdidx]

def getImportantFeatsPCA(df,n):
    std_data=StandardScaler().fit_transform(df)
    pca=PCA(n_components=5,svd_solver='full')
    pca_data=pca.fit_transform(std_data)
    importance=np.sum(np.abs(pca.components_),axis=0)
    idx = np.argpartition(-importance, n)
    orderdidx = np.argsort(-importance[idx[:n]])
    return idx[orderdidx]


n_runs=20
n_clusters=20
proportion = [5,15,30,60,90]
n_proportions=len(proportion)
scores_nmi=np.zeros([n_proportions,n_clusters])
scores_silhouette=np.zeros([n_proportions,n_clusters])
scores_fm=np.zeros([n_proportions,n_clusters])

#top100=getImportantFeats(data,100)
top100=getImportantFeatsPCA(data,100)

for r in range (n_runs):
    for i in range(n_proportions):
        n=proportion[i]
        randfeats= np.random.choice(np.arange(features), size=(100-n), replace=False)

        noise = np.random.uniform(-1.5, 1.5, size=[datapoints, 100-n])
        irrfeats=data.iloc[:, randfeats].multiply(noise)

        loop_data=pd.concat([data.iloc[:,top100[:n]],irrfeats],axis=1)
        std_data = StandardScaler().fit_transform(loop_data)

        for c in range(2,n_clusters):
            predictions=KMeans(c).fit_predict(std_data)

            nmi_score_kmeans = normalized_mutual_info_score(labels['TCGAclassstr'], predictions)
            scores_nmi[i,c] += nmi_score_kmeans

            silhouette_score_kmeans = silhouette_score(std_data, predictions)
            scores_silhouette[i,c] += silhouette_score_kmeans

            fm_score_kmeans = fowlkes_mallows_score(labels['TCGAclassstr'], predictions)
            scores_fm[i,c] += fm_score_kmeans


scores_nmi=scores_nmi/n_runs
scores_silhouette=scores_silhouette/n_runs
scores_fm=scores_fm/n_runs


ax = plt.figure().gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
for i in range(n_proportions):
    plt.plot(np.arange(2,n_clusters),scores_nmi[i,2:],label=proportion[i])
plt.xticks(np.arange(2,n_clusters,step=2))
plt.xlabel('nr of clusters')
plt.ylabel('nmi score')
plt.legend()
plt.show()

for i in range(n_proportions):
    plt.plot(np.arange(2,n_clusters),scores_silhouette[i,2:],label=proportion[i])
plt.xticks(np.arange(2,n_clusters,step=2))
plt.xlabel('nr of clusters')
plt.ylabel('silhouette score')
plt.legend()
plt.show()

for i in range(n_proportions):
    plt.plot(np.arange(2,n_clusters),scores_fm[i,2:],label=proportion[i])
plt.xticks(np.arange(2,n_clusters,step=2))
plt.xlabel('nr of clusters')
plt.ylabel('fm score')
plt.legend()
plt.show()