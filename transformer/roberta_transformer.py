import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("mariagrandury/roberta-base-finetuned-sms-spam-detection")
model = AutoModelForSequenceClassification.from_pretrained("mariagrandury/roberta-base-finetuned-sms-spam-detection")


class RobertaTransformer(BaseEstimator, TransformerMixin):

    def tweet_to_vector(self, model, tweet):
        inputs = tokenizer(tweet, return_tensors='pt')
        with torch.no_grad():
            item = model(**inputs).logits.argmax().item()
            if item:
                return 'Quality'
            else:
                return 'Spam'

    def transform(self, X, y=None):
        X_roberta = list()
        for tweet in X:
            X_roberta.append(self.tweet_to_vector(model, tweet))
        return np.stack(X_roberta, axis=0)

