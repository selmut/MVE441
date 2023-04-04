import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

n = 3

data_path = os.path.join(os.path.dirname(__file__), 'data/data.csv')
df = pd.read_csv(data_path)

# df_cat = df.astype('category').values.codes
df_cat = df.copy()

for key in df.keys():
    df_cat[key] = df[key].astype('category').values.codes
    # print(df_cat[key].value_counts(), '\n')

labels = df_cat['default_ind']
df_cat = df_cat.iloc[:, :-1]

std_data = StandardScaler().fit_transform(df_cat)

# print(std_data, np.shape(std_data))

pca = PCA(n_components=n, svd_solver='full')
pca_feats = pca.fit_transform(std_data, labels)

'''pca1 = pca_feats[:, 0]
pca2 = pca_feats[:, 1]

plt.figure()
c = ['tab:blue', 'tab:orange']
plt.scatter(pca1, pca2, c=labels, s=1, cmap=colors.ListedColormap(c))
plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+'pca_plot.png')
plt.close()

csv_path = os.path.join(os.path.dirname(__file__), 'csv/')
pca_feats.tofile(csv_path+'pca_feats.csv', sep=',', format='%10.5f')

labels.to_numpy().tofile(csv_path+'labels.csv', sep=',', format='%10.5f')'''

