import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(data):
    X = data["Tweet"]
    y = data["Type"]


    X_train, X_test, y_train, y_test = train_test_split(
        X.index,
        y,
        test_size = 0.2,
        random_state = 42
    )

    X_train = X.iloc[X_train]
    X_test = X.iloc[X_test]

    return X_train, X_test