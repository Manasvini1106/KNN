

import numpy as np
from collections import Counter


# euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# k_nearest_neighbors class

class KNN:

    # constrcutor for k nearest neighbors

    def __init__(self, k=3):
        self.y_train = None
        self.X_train = None
        self.k = k

    # fit method for machine learning

    def fit(self, X, y):
        """
        X: training samples
        y: training lables --target
        """
        self.X_train = X
        self.y_train = y

    # predict method
    def predict(self, X):
        predicted_labels = [self._helper(x) for x in X]
        return np.array(predicted_labels)

    # helper method
    def _helper(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples,labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
