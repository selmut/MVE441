import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import colors


num_points = 1000
data=np.zeros((num_points,3))

labels=np.random.randint(0,2,num_points)

xs=np.random.uniform(0,10,num_points)
ys=2*xs+labels*15
xnoise=np.random.normal(0,1,num_points)
ynoise=np.random.normal(0,1,num_points)

data[:,0]=xs+xnoise
data[:,1]=ys+ynoise


std_data = StandardScaler().fit_transform(data)
pca = PCA(n_components=1)
pca_data=pca.fit_transform(std_data)
r=pca.components_.flatten()


c = ['tab:blue', 'tab:orange']
plt.scatter(data[:,0],data[:,1], c=labels, s=5, cmap=colors.ListedColormap(c))

plt.show()
plt.scatter(pca_data*r[0],pca_data*r[1] , c=labels, s=5, cmap=colors.ListedColormap(c))
plt.show()