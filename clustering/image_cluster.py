import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from configurations import EPS, MIN_SAMPLES

class ImageCluster:
    def __init__(self):
        self.model = DBSCAN(eps=EPS,
                            min_samples=MIN_SAMPLES,
                            metric="cosine")

    def cluster_images(self, features):
        features = np.array(features)
        features = normalize(features, norm="l2")
        labels = self.model.fit_predict(features)
        return labels

