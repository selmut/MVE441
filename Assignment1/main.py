import os
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
#from Assignment1.Classifiers.gmm import GMM
from Classifiers.gmm import GMM
from Classifiers.kmeans import kMeans
from Classifiers.knn import KNN
from sklearn.model_selection import train_test_split
import seaborn as sns
from cross_valid import CV

warnings.filterwarnings("ignore")

## original data
def read_original_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data/data.csv')
    return pd.read_csv(data_path)

## subset of original data
def read_sub_data():
    subdata_path = os.path.join(os.path.dirname(__file__), 'data/subdata.csv')
    return pd.read_csv(subdata_path)
    
## generate a new csv, stratified subset of original of 1% size
def generate_sub_data(df):
    subdf = df.groupby('default_ind', group_keys=False).apply(lambda x: x.sample(frac=0.01))
    subdata_path = os.path.join(os.path.dirname(__file__), 'data/subdata.csv')
    pd.DataFrame(subdf).to_csv(subdata_path, header=True, index=False)
    print("Subset of data is generated!\n")

#generate_sub_data(read_original_data())
#df = read_original_data()
df = read_sub_data()


n_feats = len(df.keys())
n_list = list(range(1, n_feats))

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

neighbors = 5
knn = KNN(neighbors)
gmm = GMM(2)
kmeans = kMeans(2)
avg_scores_gmm = np.zeros([n_feats-1,4])
avg_scores_kmeans = np.zeros([n_feats-1,4])
avg_scores_knn = np.zeros([n_feats-1,4])

#for n in n_list:
for n in range(1,10):
    print("feat. {} out of {}..".format(n,n_feats-1))
    labels, df_cat = to_categorical(df, 'default_ind')
    pca_feats = reduce_dim(df_cat, n)

    pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats),pd.DataFrame(labels), 
                                                                                        test_size=0.2, stratify=labels)

    cv_gmm = CV(pca_test_data, pca_test_labels, 7, gmm)
    cv_kmeans = CV(pca_test_data, pca_test_labels, 7, kmeans)
    cv_knn = CV(pca_test_data, pca_test_labels, 7, knn)
    
    avg_scores_gmm[n-1,:] = cv_gmm.run_cv()
    avg_scores_kmeans[n-1,:] = cv_kmeans.run_cv()
    avg_scores_knn[n-1,:] = cv_knn.run_cv()
    
    #print(avg_scores_gmm)
    #print(avg_scores_kmeans)
    #print(cv_knn.run_cv())

print(avg_scores_gmm)
print("[Accuracy    F1    Precision    AUC]")
print("gmm max Acc: ", np.argmax(avg_scores_gmm[:,0]))
print("gmm max F1: ", np.argmax(avg_scores_gmm[:,1]))
print("gmm max Precision: ", np.argmax(avg_scores_gmm[:,2]))
print("gmm max AUC: ", np.argmax(avg_scores_gmm[:,3]))

print(avg_scores_kmeans)
print("[Accuracy    F1    Precision    AUC]")
print("kmeans max Acc: ", np.argmax(avg_scores_kmeans[:,0]))
print("kmeans max F1: ", np.argmax(avg_scores_kmeans[:,1]))
print("kmeans max Precision: ", np.argmax(avg_scores_kmeans[:,2]))
print("kmeans max AUC: ", np.argmax(avg_scores_kmeans[:,3]))

print(avg_scores_knn)
print("[Accuracy    F1    Precision    AUC]")
print("knn max Acc: ", np.argmax(avg_scores_knn[:,0]))
print("knn max F1: ", np.argmax(avg_scores_knn[:,1]))
print("knn max Precision: ", np.argmax(avg_scores_knn[:,2]))
print("knn max AUC: ", np.argmax(avg_scores_knn[:,3]))



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(avg_scores_gmm[:,0])
axs[0, 0].set_title('Accuracy')
axs[0, 1].plot(avg_scores_gmm[:,1], 'tab:orange')
axs[0, 1].set_title('F1')
axs[1, 0].plot(avg_scores_gmm[:,2], 'tab:green')
axs[1, 0].set_title('Precision')
axs[1, 1].plot(avg_scores_gmm[:,3], 'tab:red')
axs[1, 1].set_title('ROC AUC')

for ax in axs.flat:
    ax.set(xlabel='score', ylabel='pca features')

for ax in axs.flat:
    ax.label_outer()

plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+'gmm_score.png')
plt.close()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(avg_scores_kmeans[:,0])
axs[0, 0].set_title('Accuracy')
axs[0, 1].plot(avg_scores_kmeans[:,1], 'tab:orange')
axs[0, 1].set_title('F1')
axs[1, 0].plot(avg_scores_kmeans[:,2], 'tab:green')
axs[1, 0].set_title('Precision')
axs[1, 1].plot(avg_scores_kmeans[:,3], 'tab:red')
axs[1, 1].set_title('ROC AUC')

for ax in axs.flat:
    ax.set(xlabel='score', ylabel='pca features')

for ax in axs.flat:
    ax.label_outer()

plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+'kmeans_score.png')
plt.close()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(avg_scores_knn[:,0])
axs[0, 0].set_title('Accuracy')
axs[0, 1].plot(avg_scores_knn[:,1], 'tab:orange')
axs[0, 1].set_title('F1')
axs[1, 0].plot(avg_scores_knn[:,2], 'tab:green')
axs[1, 0].set_title('Precision')
axs[1, 1].plot(avg_scores_knn[:,3], 'tab:red')
axs[1, 1].set_title('ROC AUC')

for ax in axs.flat:
    ax.set(xlabel='score', ylabel='pca features')

for ax in axs.flat:
    ax.label_outer()

plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+'knn_score.png')
plt.close()

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
