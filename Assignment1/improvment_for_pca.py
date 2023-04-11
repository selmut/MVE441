import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Classifiers.gmm import GMM
from Classifiers.kmeans import kMeans
from Classifiers.knn import KNN
from sklearn.model_selection import train_test_split
from matplotlib import colors
from cross_valid import CV
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
# number of points in and dimensions
num_dims = 40
pca_feats = list(range(1,num_dims))
num_points = 1000
variance = 2

# Generate dependent data
data = np.zeros((num_points, num_dims))
data[:,0] = np.sqrt(variance)*np.random.randn(num_points)
labels = np.floor(np.random.uniform(0.5, 1.5, num_points))
data[:,-1] = labels
alpha = np.random.uniform(0,1,num_dims)

for dim in range(1,num_dims-1):
    factor = alpha[dim]
    data[:,dim] = factor*data[:,0] + (1-factor)*variance

def reduce_dim(data, n):
    std_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n, svd_solver='full')
    return pca.fit_transform(std_data)

pca_data = reduce_dim(data, pca_feats[1])

c = ['tab:blue', 'tab:orange']
plt.scatter(data[:,0], data[:,1], c=labels, s=5, cmap=colors.ListedColormap(c))
plt.show()
plt.scatter(pca_data[:,0], pca_data[:,1], c=labels, s=5, cmap=colors.ListedColormap(c))
plt.show()

gmm = GMM(2)
kmeans = kMeans(2)
knn = KNN(10)

train_data, test_data, train_labels, test_labels = train_test_split(pd.DataFrame(pca_data),pd.DataFrame(labels), 
                                                                    test_size=0.2, stratify=labels)


trd, ted, trl, tel = train_test_split(pd.DataFrame(data),pd.DataFrame(labels), 
                                        test_size=0.2, stratify=labels)
print("Scores with pca")
print(CV(test_data, test_labels, 7, gmm).run_cv())
print(CV(test_data, test_labels, 7, kmeans).run_cv())
print(CV(test_data, test_labels, 7, knn).run_cv())

print("\nScores without pca")
print(CV(ted, tel, 7, gmm).run_cv())
print(CV(ted, tel, 7, kmeans).run_cv())
print(CV(ted, tel, 7, knn).run_cv())
