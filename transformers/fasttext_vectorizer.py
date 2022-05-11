import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext.util
from fasttext import FastText

fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')


class FastTextVectorizer(BaseEstimator, TransformerMixin):

    def tweet_to_vector(self, model, tweet):
        vector = np.zeros(model.get_dimension())
        tokens = FastText.tokenize(tweet)
        for token in tokens:
            vector += model[token]
        return vector / (len(tokens) if len(tokens) else 1)

    def transform(self, X, y=None):
        X_fasttext = list()
        for tweet in X:
            X_fasttext.append(self.tweet_to_vector(ft, tweet))
        return np.stack(X_fasttext, axis=0)

    def fit(self, X, y=None, **kwargs):
        return self
