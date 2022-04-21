import pandas as pd
from data.twitter_data import load_data
from data.split_data import split_data
from data.bag_of_words import vectorize_data_with_bow
from logistic_regression import calculate_logistic_regression


if __name__ == '__main__':
    data = load_data()

    splitted_data = split_data(data)

    X_train = splitted_data[0]
    X_test = splitted_data[1]
    y_train = splitted_data[2]

    splitted_bow_data = vectorize_data_with_bow(X_train, X_test)

    X_train_bow = splitted_bow_data[0]
    X_test_bow = splitted_bow_data[1]

    print(calculate_logistic_regression(X_train_bow, y_train))

    #print(X_train)
    #print(X_test)


    #print("It aint much, but it's honest work")