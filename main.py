import logging
from operator import itemgetter

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from pathlib import Path
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


from data.split_data import split_data
from transformer.FeaturesVectorizer import FeaturesVectorizer
from transformer.SpacyPreprocessor import SpacyPreprocessor
from transformer.fasttext_vectorizer import FastTextVectorizer
from data.twitter_data import load_data
from transformer.roberta_transformer import RobertaTransformer

if __name__ == '__main__':
    data = load_data()

    X_org = data["Tweet"]
    y = data["Type"]

    splitted_data = split_data(data)

    X_train_org = splitted_data[0]
    X_test_org = splitted_data[1]
    y_train_org = splitted_data[2]
    y_test_org = splitted_data[3]

    spacy_file = Path("data/spacy.csv")
    if spacy_file.is_file():
        X_spacy = pd.read_csv("data/spacy.csv").squeeze()
        X_spacy.fillna('', inplace=True)

    else:
        X_spacy = SpacyPreprocessor().preprocess(X_org)
        X_spacy.to_csv("data/spacy.csv", index=False)


    preprocessors = [None,
                     SpacyPreprocessor()]

    classifiers = [
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(max_depth=2, random_state=0),
        KNeighborsClassifier(),
        # MultinomialNB(),
        MLPClassifier(),
        SVC(probability=True)]

    vectorizers = [
        CountVectorizer(max_features=300),
        TfidfVectorizer(max_features=300),
        FastTextVectorizer()]

    features = [False, True]

    result = []

    for prep in preprocessors:
        for vect in vectorizers:
            for feat in features:
                for clf in classifiers:

                    print("Preprocessor: " + prep.__class__.__name__)
                    print("Vectorizer: " + vect.__class__.__name__)
                    print("With features: " + str(feat))
                    print("Classifier " + clf.__class__.__name__)

                    if prep:
                        X = X_spacy
                    else:
                        X = X_org

                    if feat:
                        vector = FeatureUnion([
                            ('cv', vect),
                            ('features', FeaturesVectorizer(data))
                        ])
                    else:
                        vector = vect

                    n_splits = 5
                    accuracy = 0
                    kf = KFold(n_splits, random_state=None, shuffle=False)
                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        X_train = vector.fit_transform(X_train)
                        X_test = vector.transform(X_test)
                        predicted = clf.fit(X_train, y_train).predict(X_test)
                        report = classification_report(y_test, predicted, output_dict=True)
                        accuracy += report['accuracy']
                    accuracy = accuracy / n_splits
                    print(accuracy)
                    print()

                    result.append(
                        {
                            'prep': prep,
                            'vect': vect,
                            'feat': feat,
                            'clf': clf,
                            'best_score': accuracy
                        }
                    )

                    try:
                        spam_id = 204
                        quality_id = 203

                        c = make_pipeline(vector, clf)
                        explainer = LimeTextExplainer(class_names=['Quality', 'Spam'])
                        filename = prep.__class__.__name__ + '_' + vect.__class__.__name__ + '_' + str(feat) + '_' + clf.__class__.__name__

                        exp = explainer.explain_instance(X[spam_id], c.predict_proba, num_features=10)
                        exp.save_to_file('exp/' + filename + '_spam.html')

                        exp = explainer.explain_instance(X[quality_id], c.predict_proba, num_features=10)
                        exp.save_to_file('exp/' + filename + '_quality.html')
                    except Exception as e:
                        logging.error('Error at %s', 'lime', exc_info=e)


    print("RoBERTa Transformer")
    predicted = RobertaTransformer().transform(X_test_org)
    report = classification_report(y_test_org, predicted, output_dict=True)
    print(report['accuracy'])
    print()
    result.append(
        {
            'prep': None,
            'vect': None,
            'feat': None,
            'clf': RobertaTransformer,
            'best_score': report['accuracy']
        }
    )

    result = sorted(result, key=itemgetter('best_score'), reverse=True)

    print("Best: ")
    print("Preprocessor: " + str(result[0]['prep']))
    print("Vectorizer: " + str(result[0]['vect']))
    print("With features: " + str(result[0]['feat']))
    print("Classifier " + str(result[0]['clf']))
    print(result[0]['best_score'])