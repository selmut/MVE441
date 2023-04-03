import os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pycaret as pyc

n = 2

data_path = os.path.join(os.path.dirname(__file__), 'data/data.csv')
df = pd.read_csv(data_path)

#df['game_id'] = df['game_id']-1

data_pca = PCA(n_components=n, svd_solver='full')
# data_pca_feats = data_pca.fit_transform()

#print(df['opening_response'].value_counts())
#print(df['opening_response']=='declined')
print(df.keys())
print(df.head())
