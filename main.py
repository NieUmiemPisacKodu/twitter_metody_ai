from sklearn.metrics import classification_report

from classifiers.logistic_regression import train_logistic_regression
from classifiers.naive_bayes import train_naive_bayes
from classifiers.random_forests import train_random_forests
from data.bag_of_words import vectorize_data_with_bow
from data.split_data import split_data
from data.twitter_data import load_data


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

    print("Logistic Regression results:")
    clf_logistic_reg = train_logistic_regression(X_train_bow, y_train)
    pred_leb_logistic_reg = clf_logistic_reg.predict(X_test_bow)
    print(classification_report(y_test, pred_leb_logistic_reg))

    print("Naive Bayes results:")
    clf_naive_bayes = train_naive_bayes(X_train_bow, y_train)
    pred_leb_naive_bayes = clf_naive_bayes.predict(X_test_bow)
    print(classification_report(y_test, pred_leb_naive_bayes))

    print("Random Forests results:")
    clf_random_forests = train_random_forests(X_train_bow, y_train)
    pred_leb_random_forests = clf_random_forests.predict(X_test_bow)
    print(classification_report(y_test, pred_leb_random_forests))