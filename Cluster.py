from scipy.spatial.distance import cosine
import numpy as np
impor
class Protein:
    def __init__(self, seq, latent):
        self.seq = seq
        self.latent = np.array(latent)
    def __sub__(self, other):
        return cosine(self.latent, other.latent)


class Cluster:
    def __init__(self, seqs=None, centroid=None):
        if not seqs:
            seqs = []
        self.seqs = seqs
        if not centroid:
            centroid = []
        self.centroid = centroid
        self.len = len(seqs)

    def __add__(self, other):
        return Cluster(self.seqs + other.seqs,
                       (self.centroid * self.len + other.len * other.centroid) / (self.len + self.len))

class Tree:
    def __init__(self):
        self.clusters = []


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=30, centers=5, cluster_std=10, random_state=42)


def divisive_clustering(data, max_clusters=3):
    while len(clusters) < max_clusters:
        cluster_to_split = max(clusters, key=lambda x: len(x))
        clusters.remove(cluster_to_split)

        kmeans = KMeans(n_clusters=2, random_state=42).fit(cluster_to_split)
        cluster1 = cluster_to_split[kmeans.labels_ == 0]
        cluster2 = cluster_to_split[kmeans.labels_ == 1]

        clusters.extend([cluster1, cluster2])
    return clusters


clusters = divisive_clustering(X, max_clusters=3)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i, cluster in enumerate(clusters):
    plt.scatter(cluster[:, 0], cluster[:, 1], s=50,
                c=colors[i], label=f'Cluster {i+1}')
plt.title('Divisive Clustering Result')
plt.legend()

linked = linkage(X, method='ward')

plt.subplot(1, 2, 2)
dendrogram(linked, orientation='top',
           distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')

plt.tight_layout()
plt.show()
