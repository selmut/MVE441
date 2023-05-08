import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from Clustering.kmeans import kMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from loocv import LOOCV
import plots
import warnings

warnings.filterwarnings("ignore")

nruns = 15

df = pd.read_csv('data/Cancerdata.txt', sep="\t")
new_keys = {'lab': 'label'}
new_keys.update(dict(zip([f'Unnamed: {i}' for i in range(3000)], [f'Gene {i}' for i in range(3000)])))

df = df.rename(columns=new_keys)
print(df.head())

data = df.loc[:, 'Gene 1':'Gene 2999']
labels = df.loc[:, 'label']


def reduce_dim(df, n):
    std_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n, svd_solver='full')
    return pca.fit_transform(std_data)


pca_data = reduce_dim(data, 2)
plots.plot_pca(pca_data, labels)

pca_data = reduce_dim(data, 3)
# plots.plot_pca_3D(pca_data, labels)

labels = kMeans(3).fit_predict(pca_data)

score = silhouette_score(pca_data, labels)

# plots.plot_pca_3D(pca_data, labels)



