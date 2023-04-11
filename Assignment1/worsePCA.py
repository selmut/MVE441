import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


num_points = 1000
data=np.zeros((num_points,3))
pca_feats = list(range(1,3))

labels=np.random.randint(0,2,num_points)
xs=labels*6+np.random.normal(0,1,num_points)
ys=np.random.normal(0,10,num_points)

data[:,0]=xs
data[:,1]=ys



def reduce_dim(data, n):
    std_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n)
    return pca.fit_transform(std_data)


pca_data = reduce_dim(data, 1)


plt.scatter(data[:,0],data[:,1], c=labels, s=5)

plt.show()

plt.scatter(np.zeros(num_points), pca_data.flatten(), c=labels, s=5)
plt.show()