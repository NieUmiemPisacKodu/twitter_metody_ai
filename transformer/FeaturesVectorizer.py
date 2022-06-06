import numpy as np


class FeaturesVectorizer:

    def __init__(self, data):
        self.X = data

    def tweet_to_vector(self, i):
        return self.X.iloc[i, 1:5].to_numpy(dtype=int)

    def transform(self, X):
        X_features = list()
        for i, tweet in X.items():
            X_features.append(self.tweet_to_vector(i))
        return np.stack(X_features, axis=0)

    def fit_transform(self, X, y=None):
        return self.transform(X)