import pandas as pd


def load_data():
    df = pd.read_csv("data/train.csv")
    df.loc[[7552], ['location']] = df.loc[[7552]]['location'] + df.loc[[7552]]['Type']
    df.loc[[7552], ['Type']] = df.loc[[7552]]['Unnamed: 7']
    df.loc[[12843], ['location']] = df.loc[[12843]]['location'] + df.loc[[12843]]['Type']
    df.loc[[12843], ['Type']] = df.loc[[12843]]['Unnamed: 7']
    df = df.drop(columns=['Unnamed: 7'])
    df = df.fillna(0)
    return df