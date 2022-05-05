from sklearn.naive_bayes import GaussianNB


def train_naive_bayes(X_train, y_train):
    clf = GaussianNB().fit(X_train, y_train)
    return clf