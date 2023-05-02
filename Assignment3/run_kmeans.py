import numpy as np
import pandas as pd
from Classifiers.kmeans import kMeans
from loocv import LOOCV
import warnings

warnings.filterwarnings("ignore")

nruns = 15

df = pd.read_csv('data/Cancerdata.txt', sep="\t")
new_keys = {'lab': 'label'}
new_keys.update(dict(zip([f'Unnamed: {i}' for i in range(3000)], [f'Gene {i}' for i in range(3000)])))

df = df.rename(columns=new_keys)

data = df.loc[:, 'Gene 1':'Gene 2999']
labels = df.loc[:, 'label']

scores = np.zeros(nruns, 3)

for i in range(nruns):
#for i in range(df.shape[0]):
    kmeans = kMeans(i)
    loocv_kmeans = LOOCV(df, kmeans)
    scores[i,:] = loocv_kmeans.run()

# plot later