from sklearn.ensemble import RandomForestClassifier


def train_random_forests(X_train, y_train):
    clf = RandomForestClassifier(max_depth = 2, random_state = 0).fit(X_train, y_train)
    return clf