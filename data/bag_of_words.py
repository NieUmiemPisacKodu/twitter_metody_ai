import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def vectorize_data_with_bow(X_train, X_test):
    vectorizer = CountVectorizer(max_features=300)

    X_train_bow = vectorizer.fit_transform(X_train).toarray()
    X_test_bow = vectorizer.transform(X_test).toarray()

    pd.DataFrame(X_train_bow, columns=vectorizer.get_feature_names_out())
    return X_train_bow, X_test_bow