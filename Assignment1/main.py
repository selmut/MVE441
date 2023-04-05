import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
#from Assignment1.Classifiers.gmm import GMM
from Classifiers.gmm import GMM
from sklearn.model_selection import train_test_split
import seaborn as sns
from cross_valid import CV

data_path = os.path.join(os.path.dirname(__file__), 'data/data.csv')
df = pd.read_csv(data_path)

n_feats = len(df.keys())
n_list = list(range(1, n_feats+1))

# df_cat = df.astype('category').values.codes


def to_categorical(df, label_key):
    df_cat = df.copy()

    for key in df.keys():
        df_cat[key] = df[key].astype('category').values.codes
        # print(df_cat[key].value_counts(), '\n')

    labels = df_cat[label_key]
    df_cat = df_cat.iloc[:, :-1]
    return labels, df_cat


def reduce_dim(df, n):
    std_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n, svd_solver='full')
    return pca.fit_transform(std_data, labels)


## performs CV for all number of features in PCA with GMM

gmm = GMM(2)
avg_scores_gmm = np.zeros([n_feats,4])
for n in n_list:
    print("feat. {} out of {}".format(n,n_feats))
    labels, df_cat = to_categorical(df, 'default_ind')
    pca_feats = reduce_dim(df_cat, n)

    pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats),pd.DataFrame(labels), 
                                                                                        test_size=0.2, stratify=labels)

    cv = CV(pca_test_data, pca_test_labels, 10, gmm)
    avg_scores_gmm[n,:] = cv.run_cv()

print(avg_scores_gmm)

'''
csv_path = os.path.join(os.path.dirname(__file__), 'csv/')
pd.DataFrame(pca_train_data).to_csv(csv_path+'train_data.csv', header=False, index=False)
pd.DataFrame(pca_train_labels).to_csv(csv_path+'train_labels.csv', header=False, index=False)
pd.DataFrame(pca_test_data).to_csv(csv_path+'test_data.csv', header=False, index=False)
pd.DataFrame(pca_test_labels).to_csv(csv_path+'test_labels.csv', header=False, index=False)
'''

'''pca1 = pca_feats[:, 0]
pca2 = pca_feats[:, 1]

plt.figure()
c = ['tab:blue', 'tab:orange']
plt.scatter(pca1, pca2, c=labels, s=1, cmap=colors.ListedColormap(c))
plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+'pca_plot.png')
plt.close()

pca_feats.tofile(csv_path+'pca_feats.csv', sep=',', format='%10.5f')

labels.to_numpy().tofile(csv_path+'labels.csv', sep=',', format='%10.5f')'''
# gmm = GMM(pca_feats, 2).classify()

# sns.scatterplot(x=pca_feats[:,0],y=pca_feats[:,1],hue=gmm, s=1)
# plt.show()
