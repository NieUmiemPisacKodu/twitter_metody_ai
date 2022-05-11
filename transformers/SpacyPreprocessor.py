import spacy
import pandas as pd


class SpacyPreprocessor:

    def clean(self, tweet):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(tweet.lower())
        return ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.text.startswith('@')])


    def preprocess(self, X):
        X_test_spacy = list()
        for tweet in X:
            X_test_spacy.append(self.clean(tweet))
        return pd.Series(X_test_spacy)

