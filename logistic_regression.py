from sklearn.linear_model import LogisticRegression


def calculate_logistic_regression(X_train_bow, y_train):
    clf = LogisticRegression(random_state=0).fit(X_train_bow, y_train)
    return clf