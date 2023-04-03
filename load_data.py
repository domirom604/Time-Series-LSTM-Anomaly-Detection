import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def readDatasetPath():
    with open('zbr_danych.txt') as f:
        lines = f.readlines()
    return lines
def readDataFromFile():
    df = pd.read_csv(str.strip(readDatasetPath()[0]))
    return df

def prepare_and_load_data():
    timestamp = readDataFromFile()['Hour'].to_numpy()
    normalized_values = readDataFromFile()['value']
    isAnomaly = readDataFromFile()['Outliers'].to_numpy()
    train_size = int(len(normalized_values) *.94)
    test_size = len(normalized_values) - train_size

    train, test = normalized_values.iloc[0:train_size], normalized_values.iloc[train_size:len(normalized_values)]
    train = train.to_numpy()
    test = test.to_numpy()
    train_dataset_to_scale = train.reshape(-1,1)
    test_dataset_to_scale = test.reshape(-1, 1)
    scaler = StandardScaler()

    scaler = scaler.fit(train_dataset_to_scale)
    train = scaler.transform(train_dataset_to_scale)
    test = scaler.transform(test_dataset_to_scale)

    train = train.ravel()
    tim_train = timestamp[0:train_size]
    test = test.ravel()
    tim_test = timestamp[train_size:len(normalized_values)]

    x = tim_train, train
    x = np.array(x, dtype="float32")
    x = x.transpose()

    y = isAnomaly
    y = np.array(y, dtype="float32")
    y = y.transpose()

    x = x.tolist()[23:]
    y = y.tolist()[23:]

    new_x = []
    new_x_tmp = []
    new_y = []
    new_y_tmp = []

    xx = tim_test, test
    xx = np.array(xx, dtype="float32")
    xx = xx.transpose()
    xx = xx.tolist()[23:]
    new_xx = []
    new_xx_tmp = []

    for i,sample in enumerate(x):
        new_x_tmp.append(sample[1])
        new_y_tmp.append(y[i])
        if int(sample[0]) == 23:
            new_x.append(new_x_tmp)
            new_y.append(new_y_tmp)
            new_x_tmp = []
            new_y_tmp = []

    x = new_x
    x = np.array(x, dtype="float32")
    y = new_y
    y_train = np.array(y, dtype="float32")
    x_train = x.reshape(x.shape[0], x.shape[1], 1)

    y = isAnomaly
    y = np.array(y, dtype="float32")
    y = y.transpose()
    y = y.tolist()[4847:5159]
    new_y = []
    new_y_tmp = []
    for i,sampl in enumerate(xx):
        new_xx_tmp.append(sampl[1])
        new_y_tmp.append(y[i])
        if int(sampl[0]) == 23:
            new_xx.append(new_xx_tmp)
            new_y.append(new_y_tmp)
            new_xx_tmp = []
            new_y_tmp = []

    y = new_y
    y[0].append(0.0)
    y_test = np.array(y, dtype="float32")
    xx = new_xx
    xx[0].append(0.47927558422088623)
    xx = np.array(xx, dtype="float32")
    x_test = xx.reshape(xx.shape[0], xx.shape[1], 1)

    return x_train, x_test, y_train, y_test, scaler
