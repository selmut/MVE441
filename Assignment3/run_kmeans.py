import numpy as np
import pandas as pd

df = pd.read_csv('data/Cancerdata.txt', sep="\t")
new_keys = {'lab': 'label'}
new_keys.update(dict(zip([f'Unnamed: {i}' for i in range(3000)], [f'Gene {i}' for i in range(3000)])))

df = df.rename(columns=new_keys)

data = df.loc[:, 'Gene 1':'Gene 2999']
labels = df.loc[:, 'label']

print(data.head())
print(labels.head())


