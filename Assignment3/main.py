import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score

#https://towardsdatascience.com/7-evaluation-metrics-for-clustering-algorithms-bdc537ff54d2

import plots
import warnings

warnings.filterwarnings("ignore")

nruns = 15

df = pd.read_csv('data/Cancerdata.txt', sep="\t")
new_keys = {'lab': 'label'}
new_keys.update(dict(zip([f'Unnamed: {i}' for i in range(3000)], [f'Gene {i}' for i in range(3000)])))

df = df.rename(columns=new_keys)
#print(df.head())

data = df.loc[:, 'Gene 1':'Gene 2999']
labels = df.loc[:, 'label']


def reduce_dim(df, n, threshold):
    std_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n, svd_solver='full')

    # extracts feature with maximum variance
    var_data = df.to_numpy()
    var_feature = np.var(var_data[:,1:], axis=0)
    var_labels = var_data[:,0]
    var_thresh = 0.4*max(var_feature)

    # found index to remove and removes it after loop
    elements_to_remove = []
    for i in range(var_feature.shape[0]):
        if var_feature[i] < var_thresh:
            elements_to_remove.append(i)
    var_data = np.delete(var_data, elements_to_remove, axis=1)

    to_df = np.append(var_labels.reshape(-1, *var_labels.shape).T, var_data, axis=1)
    var_pd = pd.DataFrame(to_df)


    return [pca.fit_transform(std_data), var_data, var_pd]


[pca_data, var_data, var_pd] = reduce_dim(df, 5)

# plots.plot_pca_3D(pca_data, labels)

#n_cluster = np.arange(2,data.shape[0])
n_cluster = np.arange(2,8)
n_runs_original = np.arange(1)
n_runs_feature = np.arange(10)

score_original_data = np.zeros([3,3,n_cluster.shape[0]])
score_feature_selection = np.zeros([3,3,n_cluster.shape[0]])
score_pca = np.zeros([3,3,n_cluster.shape[0]])
print("\nCurrently computing metrics on original data")
for run in n_runs_original:
    print("Run {} out of {}...".format(run+1,n_runs_original[-1]+1))
    for cluster in n_cluster:
        print("Clusters, {} out of {}...".format(cluster,n_cluster[-1]+1))
        pred_labels_kmeans = KMeans(cluster).fit_predict(data)
        score_original_data[0,0,cluster-2] += normalized_mutual_info_score(labels, pred_labels_kmeans)
        score_original_data[1,0,cluster-2] += fowlkes_mallows_score(labels, pred_labels_kmeans)
        score_original_data[2,0,cluster-2] += silhouette_score(data, pred_labels_kmeans)

        pred_labels_gmm = GaussianMixture(n_components=cluster).fit_predict(data)
        score_original_data[0,1,cluster-2] += normalized_mutual_info_score(labels, pred_labels_gmm)
        score_original_data[1,1,cluster-2] += fowlkes_mallows_score(labels, pred_labels_gmm)
        score_original_data[2,1,cluster-2] += silhouette_score(data, pred_labels_gmm)

        pred_labels_agglo = AgglomerativeClustering(n_clusters=cluster).fit_predict(data)
        score_original_data[0,2,cluster-2] += normalized_mutual_info_score(labels, pred_labels_agglo)
        score_original_data[1,2,cluster-2] += fowlkes_mallows_score(labels, pred_labels_agglo)
        score_original_data[2,2,cluster-2] += silhouette_score(data, pred_labels_agglo)


print("\nCurrently computing metrics on feature selected data")
for run in n_runs_feature:
    print("Run {} out of {}...".format(run+1,n_runs_feature[-1]+1))
    for cluster in n_cluster:
        pred_labels_kmeans = KMeans(cluster).fit_predict(var_data)
        score_feature_selection[0,0,cluster-2] += normalized_mutual_info_score(labels, pred_labels_kmeans)
        score_feature_selection[1,0,cluster-2] += fowlkes_mallows_score(labels, pred_labels_kmeans)
        score_feature_selection[2,0,cluster-2] += silhouette_score(var_data, pred_labels_kmeans)

        pred_labels_gmm = GaussianMixture(n_components=cluster).fit_predict(var_data)
        score_feature_selection[0,1,cluster-2] += normalized_mutual_info_score(labels, pred_labels_gmm)
        score_feature_selection[1,1,cluster-2] += fowlkes_mallows_score(labels, pred_labels_gmm)
        score_feature_selection[2,1,cluster-2] += silhouette_score(var_data, pred_labels_gmm)

        pred_labels_agglo = AgglomerativeClustering(n_clusters=cluster).fit_predict(var_data)
        score_feature_selection[0,2,cluster-2] += normalized_mutual_info_score(labels, pred_labels_agglo)
        score_feature_selection[1,2,cluster-2] += fowlkes_mallows_score(labels, pred_labels_agglo)
        score_feature_selection[2,2,cluster-2] += silhouette_score(var_data, pred_labels_agglo)
        

print("\nCurrently computing metrics on pca data")
for run in n_runs_feature:
    print("Run {} out of {}...".format(run+1,n_runs_feature[-1]+1))
    for cluster in n_cluster:
        pred_labels_kmeans = KMeans(cluster).fit_predict(pca_data)
        score_pca[0,0,cluster-2] += normalized_mutual_info_score(labels, pred_labels_kmeans)
        score_pca[1,0,cluster-2] += fowlkes_mallows_score(labels, pred_labels_kmeans)
        score_pca[2,0,cluster-2] += silhouette_score(pca_data, pred_labels_kmeans)

        pred_labels_gmm = GaussianMixture(n_components=cluster).fit_predict(pca_data)
        score_pca[0,1,cluster-2] += normalized_mutual_info_score(labels, pred_labels_gmm)
        score_pca[1,1,cluster-2] += fowlkes_mallows_score(labels, pred_labels_gmm)
        score_pca[2,1,cluster-2] += silhouette_score(pca_data, pred_labels_gmm)

        pred_labels_agglo = AgglomerativeClustering(n_clusters=cluster).fit_predict(pca_data)
        score_pca[0,2,cluster-2] += normalized_mutual_info_score(labels, pred_labels_agglo)
        score_pca[1,2,cluster-2] += fowlkes_mallows_score(labels, pred_labels_agglo)
        score_pca[2,2,cluster-2] += silhouette_score(pca_data, pred_labels_agglo)


norm_original = n_runs_original[-1] + 1
norm_feature = n_runs_feature[-1] + 1
plots.plot_scores_feature_selection_combined(score_original_data/norm_original, n_cluster,'scores_vs_clusters_original_data.png')
plots.plot_scores_feature_selection_combined(score_feature_selection/norm_feature, n_cluster,'scores_vs_clusters_feat_selection_data.png')
plots.plot_scores_feature_selection_combined(score_pca/norm_feature, n_cluster,'scores_vs_clusters_pca_data.png')

#plots.plot_scores_feature_selection(score_feature_selection[0,:,:]/norm_feature, n_cluster, 'NMI')
#plots.plot_scores_feature_selection(score_feature_selection[1,:,:]/norm_feature, n_cluster, 'FM')
#plots.plot_scores_feature_selection(score_feature_selection[2,:,:]/norm_feature, n_cluster, 'Silhouette')



def resample(n_real, n_cluster, df):
    score = np.zeros([3,3,n_real])
    
    [pca_data, var_data, var_df] = reduce_dim(df,1)
    
    for n in range(n_real):
        df_sample = var_df.sample(int(0.9*len(df)), replace=True, axis=0)

        sampled_labels = df_sample.loc[:, 0]
        sampled_data = df_sample.loc[:,0:var_df.shape[1]-2]

        print(f'Realisation: {n:03d}/{n_real}')
        pred_labels_kmeans = KMeans(n_clusters=n_cluster).fit_predict(sampled_data)
        score[0,0,n] = normalized_mutual_info_score(sampled_labels, pred_labels_kmeans)
        score[1,0,n] = fowlkes_mallows_score(sampled_labels, pred_labels_kmeans)
        score[2,0,n] = silhouette_score(sampled_data, pred_labels_kmeans)

        pred_labels_gmm = GaussianMixture(n_components=n_cluster).fit_predict(sampled_data)
        score[0,1,n] = normalized_mutual_info_score(sampled_labels, pred_labels_gmm)
        score[1,1,n] = fowlkes_mallows_score(sampled_labels, pred_labels_gmm)
        score[2,1,n] = silhouette_score(sampled_data, pred_labels_gmm)

        pred_labels_agglo = AgglomerativeClustering(n_clusters=n_cluster).fit_predict(sampled_data)
        score[0,2,n] = normalized_mutual_info_score(sampled_labels, pred_labels_agglo)
        score[1,2,n] = fowlkes_mallows_score(sampled_labels, pred_labels_agglo)
        score[2,2,n] = silhouette_score(sampled_data, pred_labels_agglo)

    return score

n_real = 500
score_histogram = resample(n_real, 3, df)

plt.figure()
plt.hist(score_histogram[0,0,:], bins=15)
plt.savefig('img/hist_var_nmi_kmeans.png')
plt.close()

plt.figure()
plt.hist(score_histogram[1,0,:], bins=15)
plt.savefig('img/hist_var_fm_kmeans.png')
plt.close()

plt.figure()
plt.hist(score_histogram[2,0,:], bins=15)
plt.savefig('img/hist_var_sil_kmeans.png')
plt.close()

plt.figure()
plt.hist(score_histogram[0,1,:], bins=15)
plt.savefig('img/hist_var_nmi_gmm.png')
plt.close()

plt.figure()
plt.hist(score_histogram[1,1,:], bins=15)
plt.savefig('img/hist_var_fm_gmm.png')
plt.close()

plt.figure()
plt.hist(score_histogram[2,1,:], bins=15)
plt.savefig('img/hist_var_sil_gmm.png')
plt.close()

plt.figure()
plt.hist(score_histogram[0,2,:], bins=15)
plt.savefig('img/hist_var_nmi_agglo.png')
plt.close()

plt.figure()
plt.hist(score_histogram[1,2,:], bins=15)
plt.savefig('img/hist_var_fm_agglo.png')
plt.close()

plt.figure()
plt.hist(score_histogram[2,2,:], bins=15)
plt.savefig('img/hist_var_sil_agglo.png')
plt.close()


def score_heatmap(data, true_labels, n_thresholds, n_clusters):
    scores_silhouette = np.zeros((3, len(n_thresholds), len(n_clusters)))
    scores_nmi = np.zeros((3, len(n_thresholds), len(n_clusters)))
    scores_fm = np.zeros((3, len(n_thresholds), len(n_clusters)))

    for i, threshold in enumerate(n_thresholds):
        pca_data, var_data = reduce_dim(data, 1, threshold)
        print(f'Threshold {threshold}')

        for j, c in enumerate(n_clusters):
            # kMeans --------------------------------------------------------------------------------------------------
            predicted_labels_kmeans = KMeans(c).fit_predict(var_data)

            nmi_score_kmeans = normalized_mutual_info_score(true_labels, predicted_labels_kmeans)
            scores_nmi[0, i, j] = nmi_score_kmeans

            silhouette_score_kmeans = silhouette_score(var_data, predicted_labels_kmeans)
            scores_silhouette[0, i, j] = silhouette_score_kmeans

            fm_score_kmeans = fowlkes_mallows_score(true_labels, predicted_labels_kmeans)
            scores_fm[0, i, j] = fm_score_kmeans

            # GMM -----------------------------------------------------------------------------------------------------
            predicted_labels_gmm = GaussianMixture(c).fit_predict(var_data)

            nmi_score_gmm = normalized_mutual_info_score(true_labels, predicted_labels_gmm)
            scores_nmi[1, i, j] = nmi_score_gmm

            silhouette_score_gmm = silhouette_score(var_data, predicted_labels_gmm)
            scores_silhouette[1, i, j] = silhouette_score_gmm

            fm_score_gmm = fowlkes_mallows_score(true_labels, predicted_labels_gmm)
            scores_fm[1, i, j] = fm_score_gmm

            # Agglomerative -------------------------------------------------------------------------------------------
            predicted_labels_agglo = AgglomerativeClustering(c).fit_predict(var_data)

            nmi_score_agglo = normalized_mutual_info_score(true_labels, predicted_labels_agglo)
            scores_nmi[2, i, j] = nmi_score_agglo

            silhouette_score_agglo = silhouette_score(var_data, predicted_labels_agglo)
            scores_silhouette[2, i, j] = silhouette_score_agglo

            fm_score_agglo = fowlkes_mallows_score(true_labels, predicted_labels_agglo)
            scores_fm[2, i, j] = fm_score_agglo

    return scores_silhouette, scores_nmi, scores_fm


n_real = 1
n_thresholds = np.arange(0, 10)*0.1  # len(data))
print(n_thresholds)
n_clusters = np.arange(2, 20)  # len(data))

scores_silhouette = np.zeros((3, len(n_thresholds), len(n_clusters)))
scores_nmi = np.zeros((3, len(n_thresholds), len(n_clusters)))
scores_fm = np.zeros((3, len(n_thresholds), len(n_clusters)))

for n in range(n_real):
    print(f'\nRealisation nr. {n+1}/{n_real}')
    tmp_scores_silhouette, tmp_scores_nmi, tmp_scores_fm = score_heatmap(data, labels, n_thresholds, n_clusters)
    scores_silhouette = (scores_silhouette + tmp_scores_silhouette)
    scores_nmi = (scores_nmi+tmp_scores_nmi)
    scores_fm = (scores_fm+tmp_scores_fm)


plots.plot_dim_vs_clusters(scores_silhouette[1, :, :], n_thresholds, n_clusters, 'silhouette_heatmap_gmm.png')
plots.plot_dim_vs_clusters(scores_silhouette[2, :, :], n_thresholds, n_clusters, 'silhouette_heatmap_agglo.png')
plots.plot_dim_vs_clusters(scores_silhouette[0, :, :], n_thresholds, n_clusters, 'silhouette_heatmap_kmeans.png')

plots.plot_dim_vs_clusters(scores_nmi[1, :, :], n_thresholds, n_clusters, 'nmi_heatmap_gmm.png')
plots.plot_dim_vs_clusters(scores_nmi[2, :, :], n_thresholds, n_clusters, 'nmi_heatmap_agglo.png')
plots.plot_dim_vs_clusters(scores_nmi[0, :, :], n_thresholds, n_clusters, 'nmi_heatmap_kmeans.png')

plots.plot_dim_vs_clusters(scores_fm[1, :, :], n_thresholds, n_clusters, 'fm_heatmap_gmm.png')
plots.plot_dim_vs_clusters(scores_fm[2, :, :], n_thresholds, n_clusters, 'fm_heatmap_agglo.png')
plots.plot_dim_vs_clusters(scores_fm[0, :, :], n_thresholds, n_clusters, 'fm_heatmap_kmeans.png')



