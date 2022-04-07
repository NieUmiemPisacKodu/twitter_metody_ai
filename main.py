import pandas as pd
from data.twitter_data import load_data
from data.split_data import split_data
from data.bag_of_words import vectorize_data_with_bow


if __name__ == '__main__':
    data = load_data()

    splitted_data = split_data(data)

    X_train = splitted_data[0]
    X_test = splitted_data[1]

    vectorize_data_with_bow(X_train, X_test)

    print(X_train)
    print(X_test)


    print("It aint much, but it's honest work")