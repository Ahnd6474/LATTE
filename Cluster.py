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