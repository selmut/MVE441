import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score
import plots
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data/Cancerdata.txt', sep="\t")
new_keys = {'lab': 'label'}
new_keys.update(dict(zip([f'Unnamed: {i}' for i in range(3000)], [f'Gene {i}' for i in range(3000)])))

df = df.rename(columns=new_keys)
data = df.loc[:, 'Gene 1':'Gene 2999']
labels = df.loc[:, 'label']


def reduce_dim(df, n):
    std_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n, svd_solver='full')
    return pca.fit_transform(std_data)


def score_heatmap(data, true_labels, n_axes, n_clusters):
    scores_silhouette = np.zeros((3, len(n_axes), len(n_clusters)))
    scores_nmi = np.zeros((3, len(n_axes), len(n_clusters)))
    scores_fm = np.zeros((3, len(n_axes), len(n_clusters)))

    for i, ax in enumerate(n_axes):
        pca_data = reduce_dim(data, ax)
        print(f'PCA-dim {ax}')

        for j, c in enumerate(n_clusters):
            # kMeans --------------------------------------------------------------------------------------------------
            predicted_labels_kmeans = KMeans(c).fit_predict(pca_data)

            nmi_score_kmeans = normalized_mutual_info_score(true_labels, predicted_labels_kmeans)
            scores_nmi[0, i, j] = nmi_score_kmeans

            silhouette_score_kmeans = silhouette_score(pca_data, predicted_labels_kmeans)
            scores_silhouette[0, i, j] = silhouette_score_kmeans

            fm_score_kmeans = fowlkes_mallows_score(true_labels, predicted_labels_kmeans)
            scores_fm[0, i, j] = fm_score_kmeans

            # GMM -----------------------------------------------------------------------------------------------------
            predicted_labels_gmm = GaussianMixture(c).fit_predict(pca_data)

            nmi_score_gmm = normalized_mutual_info_score(true_labels, predicted_labels_gmm)
            scores_nmi[1, i, j] = nmi_score_gmm

            silhouette_score_gmm = silhouette_score(pca_data, predicted_labels_gmm)
            scores_silhouette[1, i, j] = silhouette_score_gmm

            fm_score_gmm = fowlkes_mallows_score(true_labels, predicted_labels_gmm)
            scores_fm[1, i, j] = fm_score_gmm

            # Agglomerative -------------------------------------------------------------------------------------------
            predicted_labels_agglo = AgglomerativeClustering(c).fit_predict(pca_data)

            nmi_score_agglo = normalized_mutual_info_score(true_labels, predicted_labels_agglo)
            scores_nmi[2, i, j] = nmi_score_agglo

            silhouette_score_agglo = silhouette_score(pca_data, predicted_labels_agglo)
            scores_silhouette[2, i, j] = silhouette_score_agglo

            fm_score_agglo = fowlkes_mallows_score(true_labels, predicted_labels_agglo)
            scores_fm[2, i, j] = fm_score_agglo

    return scores_silhouette, scores_nmi, scores_fm


n_real = 1

n_axes = np.arange(1, 20)  # len(data))
n_clusters = np.arange(2, 20)  # len(data))

scores_silhouette = np.zeros((3, len(n_axes), len(n_clusters)))
scores_nmi = np.zeros((3, len(n_axes), len(n_clusters)))
scores_fm = np.zeros((3, len(n_axes), len(n_clusters)))

for n in range(n_real):
    print(f'\nRealisation nr. {n+1}/{n_real}')
    tmp_scores_silhouette, tmp_scores_nmi, tmp_scores_fm = score_heatmap(data, labels, n_axes, n_clusters)
    scores_silhouette = (scores_silhouette + tmp_scores_silhouette)/2
    scores_nmi = (scores_nmi+tmp_scores_nmi)/2
    scores_fm = (scores_fm+tmp_scores_fm)/2


plots.plot_dim_vs_clusters(scores_silhouette[1, :, :], n_axes, n_clusters, 'silhouette_heatmap_gmm.png')
plots.plot_dim_vs_clusters(scores_silhouette[2, :, :], n_axes, n_clusters, 'silhouette_heatmap_agglo.png')
plots.plot_dim_vs_clusters(scores_silhouette[0, :, :], n_axes, n_clusters, 'silhouette_heatmap_kmeans.png')

plots.plot_dim_vs_clusters(scores_nmi[1, :, :], n_axes, n_clusters, 'nmi_heatmap_gmm.png')
plots.plot_dim_vs_clusters(scores_nmi[2, :, :], n_axes, n_clusters, 'nmi_heatmap_agglo.png')
plots.plot_dim_vs_clusters(scores_nmi[0, :, :], n_axes, n_clusters, 'nmi_heatmap_kmeans.png')

plots.plot_dim_vs_clusters(scores_fm[1, :, :], n_axes, n_clusters, 'fm_heatmap_gmm.png')
plots.plot_dim_vs_clusters(scores_fm[2, :, :], n_axes, n_clusters, 'fm_heatmap_agglo.png')
plots.plot_dim_vs_clusters(scores_fm[0, :, :], n_axes, n_clusters, 'fm_heatmap_kmeans.png')


def score_for_chosen_pca_dim(n_clusters, data, pca_dim, true_labels):
    chosen_dim_silhouette = np.zeros((3, len(n_clusters)))
    chosen_dim_nmi = np.zeros((3, len(n_clusters)))
    chosen_dim_fm = np.zeros((3, len(n_clusters)))

    for i, c in enumerate(n_clusters):
        pca_data = reduce_dim(data, pca_dim)

        # kMeans
        predicted_labels_kmeans = KMeans(c).fit_predict(pca_data)

        nmi_kmeans = normalized_mutual_info_score(true_labels, predicted_labels_kmeans)
        chosen_dim_nmi[0, i] = nmi_kmeans

        silhouette_kmeans = silhouette_score(pca_data, predicted_labels_kmeans)
        chosen_dim_silhouette[0, i] = silhouette_kmeans

        fm_kmeans = fowlkes_mallows_score(true_labels, predicted_labels_kmeans)
        chosen_dim_fm[0, i] = fm_kmeans

        # GMM
        predicted_labels_gmm = GaussianMixture(c).fit_predict(pca_data)

        nmi_gmm = normalized_mutual_info_score(true_labels, predicted_labels_gmm)
        chosen_dim_nmi[1, i] = nmi_gmm

        silhouette_gmm = silhouette_score(pca_data, predicted_labels_gmm)
        chosen_dim_silhouette[1, i] = silhouette_gmm

        fm_gmm = fowlkes_mallows_score(true_labels, predicted_labels_gmm)
        chosen_dim_fm[1, i] = fm_gmm

        # Agglomerative
        predicted_labels_agglo = AgglomerativeClustering(c).fit_predict(pca_data)

        nmi_agglo = normalized_mutual_info_score(true_labels, predicted_labels_agglo)
        chosen_dim_nmi[2, i] = nmi_agglo

        silhouette_agglo = silhouette_score(pca_data, predicted_labels_agglo)
        chosen_dim_silhouette[2, i] = silhouette_agglo

        fm_agglo = fowlkes_mallows_score(true_labels, predicted_labels_agglo)
        chosen_dim_fm[2, i] = fm_agglo

    return chosen_dim_nmi, chosen_dim_silhouette, chosen_dim_fm


n_clusters = np.arange(2, 15)
pca_dim = 5

chosen_dim_silhouette = np.zeros((3, len(n_clusters)))
chosen_dim_nmi = np.zeros((3, len(n_clusters)))
chosen_dim_fm = np.zeros((3, len(n_clusters)))

n_real = 30
'''for n in range(n_real):
    tmp_chosen_dim_nmi, tmp_chosen_dim_silhouette, tmp_chosen_dim_fm = score_for_chosen_pca_dim(n_clusters, data, pca_dim, labels)

    chosen_dim_nmi = (chosen_dim_nmi+tmp_chosen_dim_nmi)/2
    chosen_dim_silhouette = (chosen_dim_silhouette+tmp_chosen_dim_silhouette)/2
    chosen_dim_fm = (chosen_dim_fm+tmp_chosen_dim_fm)/2

print(f'NMI (class overlap) kMeans: {chosen_dim_nmi[0, 1]}')
print(f'NMI (class overlap) GMM: {chosen_dim_nmi[1, 1]}')
print(f'NMI (class overlap) Agglomerative: {chosen_dim_nmi[2, 1]} ')

plots.plot_scores_vs_clusters(chosen_dim_silhouette, n_clusters, f'pca{pca_dim}_silhouette.png', 'Silhouette score')
plots.plot_scores_vs_clusters(chosen_dim_nmi, n_clusters, f'pca{pca_dim}_nmi.png', 'NMI')
plots.plot_scores_vs_clusters(chosen_dim_fm, n_clusters, f'pca{pca_dim}_fm.png', 'Fowlkes-Mallows score')'''


def resample(n_real, n_clusters, df, pca_dim):
    chosen_dim_nmi = np.zeros((n_real, 3, len(n_clusters)))
    chosen_dim_silhouette = np.zeros((n_real, 3, len(n_clusters)))
    chosen_dim_fm = np.zeros((n_real, 3, len(n_clusters)))

    for n in range(n_real):
        df_sample = df.sample(int(0.9*len(df)), replace=True, axis=0)

        sampled_data = df_sample.loc[:, 'Gene 1':'Gene 2999']
        sampled_labels = df_sample.loc[:, 'label']

        print(f'Realisation: {n:03d}/{n_real}')
        tmp_chosen_dim_nmi, tmp_chosen_dim_silhouette, tmp_chosen_dim_fm = score_for_chosen_pca_dim(n_clusters, sampled_data,
                                                                                                    pca_dim, sampled_labels)

        chosen_dim_nmi[n, :, :] = tmp_chosen_dim_nmi
        chosen_dim_silhouette[n, :, :] = tmp_chosen_dim_silhouette
        chosen_dim_fm[n, :, :] = tmp_chosen_dim_fm

    return chosen_dim_nmi, chosen_dim_silhouette, chosen_dim_fm


n_real = 500
chosen_dim_nmi, chosen_dim_silhouette, chosen_dim_fm = resample(n_real, n_clusters, df, pca_dim)

print(chosen_dim_nmi.shape)

plots.plot_scores_hist(chosen_dim_nmi, 0, 3, 'nmi_hist_kmeans.png')
plots.plot_scores_hist(chosen_dim_nmi, 1, 3, 'nmi_hist_gmm.png')
plots.plot_scores_hist(chosen_dim_nmi, 2, 3, 'nmi_hist_agglomerative.png')

plots.plot_scores_hist(chosen_dim_silhouette, 0, 3, 'silhouette_hist_kmeans.png')
plots.plot_scores_hist(chosen_dim_silhouette, 1, 3, 'silhouette_hist_gmm.png')
plots.plot_scores_hist(chosen_dim_silhouette, 2, 3, 'silhouette_hist_agglomerative.png')

plots.plot_scores_hist(chosen_dim_fm, 0, 3, 'fm_hist_kmeans.png')
plots.plot_scores_hist(chosen_dim_fm, 1, 3, 'fm_hist_gmm.png')
plots.plot_scores_hist(chosen_dim_fm, 2, 3, 'fm_hist_agglomerative.png')



