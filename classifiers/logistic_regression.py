from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(random_state = 0).fit(X_train, y_train)
    return clf