from operator import itemgetter

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC

from transformers.FeaturesVectorizer import FeaturesVectorizer
from transformers.SpacyPreprocessor import SpacyPreprocessor
from transformers.fasttext_vectorizer import FastTextVectorizer
from data.split_data import split_data
from data.twitter_data import load_data


if __name__ == '__main__':
    data = load_data()

    splitted_data = split_data(data)

    X_train_org = splitted_data[0]
    X_test_org = splitted_data[1]
    y_train = splitted_data[2]
    y_test = splitted_data[3]

    X_train_spacy = SpacyPreprocessor().preprocess(X_train_org)
    X_test_spacy = SpacyPreprocessor().preprocess(X_test_org)

    preprocessors = [None, SpacyPreprocessor()]

    classifiers = [LogisticRegression(),
                   RandomForestClassifier(max_depth=2, random_state=0),
                   KNeighborsClassifier(),
                   # MultinomialNB(),
                   MLPClassifier(),
                   SVC()
                   ]
    vectorizers = [CountVectorizer(max_features=300),
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
                        X_train = X_train_spacy
                        X_test = X_test_spacy
                    else:
                        X_train = X_train_org
                        X_test = X_test_org

                    if feat:
                        vector = FeatureUnion([
                            ('cv', vect),
                            ('features', FeaturesVectorizer(data))
                        ])
                    else:
                        vector = vect

                    X_train = vector.fit_transform(X_train)
                    X_test = vector.transform(X_test)
                    predicted = clf.fit(X_train, y_train).predict(X_test)
                    report = classification_report(y_test, predicted, output_dict=True)
                    print(report['accuracy'])
                    print()

                    result.append(
                        {
                            'prep': prep,
                            'vect': vect,
                            'feat': feat,
                            'clf': clf,
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

