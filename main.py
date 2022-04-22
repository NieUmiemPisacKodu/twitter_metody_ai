import pandas as pd
from data.twitter_data import load_data
from data.split_data import split_data
from data.bag_of_words import vectorize_data_with_bow
from logistic_regression import train_logistic_regression
from sklearn.metrics import classification_report


if __name__ == '__main__':
    data = load_data()

    splitted_data = split_data(data)

    X_train = splitted_data[0]
    X_test = splitted_data[1]
    y_train = splitted_data[2]
    y_test = splitted_data[3]

    splitted_bow_data = vectorize_data_with_bow(X_train, X_test)

    X_train_bow = splitted_bow_data[0]
    X_test_bow = splitted_bow_data[1]

    clf = train_logistic_regression(X_train_bow, y_train)

    pred_leb = clf.predict(X_test_bow)
    print(classification_report(y_test, pred_leb))

    #print(X_train)
    #print(X_test)


    #print("It aint much, but it's honest work")