import os
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve
import numpy as np
from Classifiers.gmm import GMM
from Classifiers.kmeans import kMeans
from Classifiers.knn import KNN
from sklearn.model_selection import train_test_split
import seaborn as sns
from cross_valid import CV

warnings.filterwarnings("ignore")

# original data
def read_original_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data/data.csv')
    return pd.read_csv(data_path)

# subset of original data
def read_sub_data():
    subdata_path = os.path.join(os.path.dirname(__file__), 'data/subdata.csv')
    return pd.read_csv(subdata_path)
    
# generate a new csv, stratified subset of original of 1% size
def generate_sub_data(df):
    subdf = df.groupby('default_ind', group_keys=False).apply(lambda x: x.sample(frac=0.01))
    subdata_path = os.path.join(os.path.dirname(__file__), 'data/subdata.csv')
    pd.DataFrame(subdf).to_csv(subdata_path, header=True, index=False)
    print("Subset of data is generated!\n")


# generate_sub_data(read_original_data())
df = read_original_data()
df_sub = read_sub_data()


n_feats = len(df.keys())

# df_cat = df.astype('category').values.codes

# converts our dataframe to categorical dataframe
def to_categorical(df, label_key):
    df_cat = df.copy()

    for key in df.keys():
        df_cat[key] = df[key].astype('category').values.codes
        # print(df_cat[key].value_counts(), '\n')

    labels = df_cat[label_key]
    df_cat = df_cat.iloc[:, :-1]
    return labels, df_cat

# normalizes and centers the data and then performs pca
def reduce_dim(df, n):
    std_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n, svd_solver='full')
    return pca.fit_transform(std_data)
    # return pca.fit_transform(std_data, labels)

def metrics(predicted_labels, true_labels):
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)  # good score to consider since it is adjusted to imbalanced data sets, which we have
    precision = precision_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    return conf_mat, [acc, f1, precision, roc_auc], [fpr, tpr, thresholds]


# saves plots of the scores for the different classifiers
def plot_scores(scores, classifier):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(scores[:,0]), axs[0, 0].set_title('Accuracy')
    axs[0, 1].plot(scores[:,1], 'tab:orange'), axs[0, 1].set_title('F1')
    axs[1, 0].plot(scores[:,2], 'tab:green'), axs[1, 0].set_title('Precision')
    axs[1, 1].plot(scores[:,3], 'tab:red'), axs[1, 1].set_title('ROC AUC')
    
    axs[0,0].set(ylabel='score')
    axs[1,0].set(xlabel='pca features', ylabel='score')
    axs[1,1].set(xlabel='pca features')
    
    axs[0,0].get_xaxis().set_visible(False)
    axs[0,1].get_xaxis().set_visible(False)
    
    fig.suptitle("Scores when classifing with "+classifier, fontsize = 16)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+classifier+'_sub_score.png')
    plt.close()


# runs the pca reduction, performs CV, classifies and then plots
def run_classification_each_pca_dim(df, n_feats):
    # constructs all classifiers
    n_list = list(range(1, n_feats))
    gmm = GMM(2)
    kmeans = kMeans(2)
    neighbors = 10
    knn = KNN(neighbors)

    # stores average scores for each classifier for each number of pca features 
    avg_scores_gmm = np.zeros([n_feats-1,4])
    avg_scores_kmeans = np.zeros([n_feats-1,4])
    avg_scores_knn = np.zeros([n_feats-1,4])

    # loops through all possible number of features of pca
    for n in n_list:
    # for n in range(1,2):
        print("feat. {} out of {}...".format(n,n_feats-1))

        # converts the data to categorical data and performs pca 
        labels, df_cat = to_categorical(df, 'default_ind')
        pca_feats = reduce_dim(df_cat, n)

        # TODO use split data correctly, and get metrics using test_data/labels
        # splits into test and train data
        pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats),pd.DataFrame(labels), 
                                                                                            test_size=0.2, stratify=labels)

        # performs CV for each classifier
        cv_gmm = CV(pca_test_data, pca_test_labels, 7, gmm)
        cv_kmeans = CV(pca_test_data, pca_test_labels, 7, kmeans)
        cv_knn = CV(pca_test_data, pca_test_labels, 7, knn)

        # stores the average score of CV
        avg_scores_gmm[n-1,:] = cv_gmm.run_cv()
        avg_scores_kmeans[n-1,:] = cv_kmeans.run_cv()
        avg_scores_knn[n-1,:] = cv_knn.run_cv()
        #print(avg_scores_knn[n-1,:])

    plot_scores(avg_scores_gmm, 'GMM')
    plot_scores(avg_scores_kmeans, 'KMeans')
    plot_scores(avg_scores_knn, 'KNN')

    # prints index of maximum scores
    print("gmm max Acc: ", np.argmax(avg_scores_gmm[:,0]))
    print("gmm max F1: ", np.argmax(avg_scores_gmm[:,1]))
    print("gmm max Precision: ", np.argmax(avg_scores_gmm[:,2]))
    print("gmm max AUC: ", np.argmax(avg_scores_gmm[:,3]))

    print("kmeans max Acc: ", np.argmax(avg_scores_kmeans[:,0]))
    print("kmeans max F1: ", np.argmax(avg_scores_kmeans[:,1]))
    print("kmeans max Precision: ", np.argmax(avg_scores_kmeans[:,2]))
    print("kmeans max AUC: ", np.argmax(avg_scores_kmeans[:,3]))

    print("knn max Acc: ", np.argmax(avg_scores_knn[:,0]))
    print("knn max F1: ", np.argmax(avg_scores_knn[:,1]))
    print("knn max Precision: ", np.argmax(avg_scores_knn[:,2]))
    print("knn max AUC: ", np.argmax(avg_scores_knn[:,3]))
    print("[Accuracy    F1    Precision    AUC]")


# Run our 
def run_classification(train_data, test_data, train_labels, test_labels):
    # print("starting classification on full data set...")
    gmm = GMM(2)
    kmeans = kMeans(2)
    neighbors = 10
    nRuns = 5
    knn = KNN(neighbors)
    
    print('Starting GMM...\n')
    scores_matrix = np.zeros((nRuns, 4))

    for n in range(nRuns):
        print(f'GMM run nr. {n}...')
        gmm_model = gmm.fit_data(train_data, train_labels)
        gmm_predictions = gmm.predict(test_data, test_labels, gmm_model)
        gmm_conf_mat, gmm_scores, gmm_rates = metrics(gmm_predictions, test_labels)

        scores_matrix[n, :] = gmm_scores

    avg_gmm_scores = np.sum(scores_matrix, axis=0)/nRuns

    print('Starting kMeans...\n')
    scores_matrix = np.zeros((nRuns, 4))

    for n in range(nRuns):
        print(f'kMeans run nr. {n}...')
        kmeans_model = kmeans.fit_data(train_data, train_labels)
        kmeans_predictions = kmeans.predict(test_data, test_labels, kmeans_model)
        kmeans_conf_mat, kmeans_scores, kmeans_rates = metrics(kmeans_predictions, test_labels)

        scores_matrix[n, :] = kmeans_scores

    avg_kmeans_scores = np.sum(scores_matrix, axis=0)/nRuns

    '''print('Starting KNN...\n')
    scores_matrix = np.zeros((nRuns, 4))

    for n in range(nRuns):
        print(f'KNN run nr. {n}...')
        knn_model = knn.fit_data(train_data, train_labels)
        knn_predictions = knn.predict(test_data, test_labels, knn_model)
        knn_conf_mat, knn_scores, knn_rates = metrics(knn_predictions, test_labels)

        scores_matrix[n, :] = knn_scores

    avg_knn_scores = np.sum(scores_matrix, axis=0)/nRuns'''
    avg_knn_scores = np.zeros(4)

    # stores the average score of CV
    print("GMM scores on dataset: ", avg_gmm_scores)
    print("\nKMeans scores on dataset: ", avg_kmeans_scores)
    print("\nKNN scores on dataset: ", avg_knn_scores)

    return avg_gmm_scores, avg_kmeans_scores, avg_knn_scores


# converts the data to categorical data
labels, df_cat = to_categorical(df, 'default_ind')
# splits into test and train data
train_data, test_data, train_labels, test_labels = train_test_split(pd.DataFrame(df_cat), pd.DataFrame(labels),
                                                                    test_size=0.2, stratify=labels)
# run_classification(train_data, test_data, train_labels, test_labels)

pca_dim = np.arange(10)+1
gmm_scores = np.zeros((10, 4))
kmeans_scores = np.zeros((10, 4))
knn_scores = np.zeros((10, 4))

for i in range(len(pca_dim)):
    print(f'\nPCA dimension {pca_dim[i]}...')
    # converts the data to categorical data and performs pca
    pca_feats = reduce_dim(df_cat, pca_dim[i])

    # splits into test and train data
    pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats),
                                                                                    pd.DataFrame(labels),
                                                                                    test_size=0.2, stratify=labels)
    avg_gmm_scores, avg_kmeans_scores, avg_knn_scores = run_classification(pca_train_data, pca_test_data, pca_train_labels, pca_test_labels)
    gmm_scores[i, :] = avg_gmm_scores
    kmeans_scores[i, :] = avg_kmeans_scores
    knn_scores[i, :] = avg_knn_scores

print(gmm_scores)   # dim 16 is optimal
print(kmeans_scores)   # dim 23 is optimal
print(knn_scores)   # dim 9 is optimal

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
