import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), 'csv/')
pca_feats = pd.read_csv(csv_path+'pca_feats.csv').to_numpy()
labels = pd.read_csv(csv_path+'labels.csv').to_numpy()

print(np.shape(pca_feats))

'''pca1 = pca_feats[:, 0]
pca2 = pca_feats[:, 1]

plt.figure()
c = ['tab:blue', 'tab:orange']
plt.scatter(pca1, pca2, c=labels, s=1, cmap=colors.ListedColormap(c))
plt.show()'''
