from operator import itemgetter

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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

    preprocessors = [None, SpacyPreprocessor()]

    classifiers = [LogisticRegression(),
                   RandomForestClassifier(max_depth=2, random_state=0),
                   KNeighborsClassifier(),
                   # GaussianNB()
                   # MLPClassifier(),
                   SVC()]
    vectorizers = [CountVectorizer(max_features=300),
                   TfidfVectorizer(max_features=300),
                   FastTextVectorizer()]

    result = []

    for prep in preprocessors:
        for vect in vectorizers:
            for clf in classifiers:

                pipeline = Pipeline([
                    ('pre', prep),
                    ('vect', vect),
                    ('clf', clf)
                ])

                print(prep.__class__.__name__)
                print(vect.__class__.__name__)
                print(clf.__class__.__name__)

                if prep:
                    X_train = SpacyPreprocessor().preprocess(X_train_org)
                    X_test = SpacyPreprocessor().preprocess(X_test_org)
                else:
                    X_train = X_train_org
                    X_test = X_test_org

                predicted = pipeline.fit(X_train, y_train).predict(X_test)
                print(classification_report(y_test, predicted))

                #cross validation using grid search
                grid_search = GridSearchCV(pipeline, param_grid={}, cv=4, n_jobs=4, verbose=1)
                grid_search.fit(X_train, y_train)

                result.append(
                        {
                            'grid': grid_search,
                            'best_score': grid_search.best_score_
                        }
                    )

    result = sorted(result, key=itemgetter('best_score'), reverse=True)

    grid = result[0]['grid']
    print("Best: ", grid.best_estimator_)
    print(classification_report(y_test, grid.best_estimator_.predict(X_test)))

