import datetime
import seaborn as sns
from keras.layers import Input, LSTM, Dropout, RepeatVector, Dense, TimeDistributed
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping

import tensorflow as tf
from keras.models import Sequential
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from load_data import *
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

# funkcja trenująca sieć enkodera-dekodera z warstwą dropout
def trainEncoderDecoder(X_train, y_train, nUnits, dropout, batch_size, epochs):
    # wejście i warstwa enkodera
    model = Sequential()
    model.add(LSTM(units=nUnits, input_shape= (X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(rate=dropout))
    model.add(RepeatVector(n = X_train.shape[1]))
    # wejście i warstwa dekodera
    model.add(LSTM(units=nUnits, return_sequences=True))
    model.add(Dropout(rate=dropout))
    #output sieci
    model.add(TimeDistributed(Dense(units=X_train.shape[2])))
    model.compile(optimizer='adam', loss='mae')

    model.fit(X_train, y_train,
              shuffle=False,
              batch_size=batch_size,
              validation_split=0.1,
              epochs=epochs)
    return model

def prepareDataToPresent():
    readed = pd.read_csv(str.strip(readDatasetPath()[1]),
        sep=';', parse_dates=['timestamp'], index_col=['timestamp'])
    train_size = int(len(readed) * .94)
    test_size = len(readed) - train_size

    train, test = readed.iloc[0:train_size], readed.iloc[train_size:len(readed)]
    return train, test

def changeDimension(x,y):
    X_train_pred_1D = []
    X_train_1D = []
    ytr = []

    for i in range(x):
        for j in range(y):
            X_train_1D.append(X_train[j][i])
            X_train_pred_1D.append(X_train_pred[j][i])
            ytr.append(y_train[j][i])

    X_train_pred_1D = np.array(X_train_pred_1D)
    X_train_1D = np.array(X_train_1D)
    return X_train_pred_1D, X_train_1D, ytr

def anomalyDetectionBasedTreshold():
    test_score_df = pd.DataFrame(index=train[3:].index)
    test_score_df['loss'] = train_mae_los
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    anomalies = test_score_df[test_score_df.anomaly == True]
    test_score_df['value'] = train[3:].value
    return test_score_df, anomalies

def xANDyPreparation(size):
    arr = []
    val = []

    for i in range(size):
        arr.append(i)
        val.append(float(train[3:].value[i]))
    return arr, val

def denormalizeDataSet(dataset):
    dat = dataset.to_numpy()
    datReshaped = dat.reshape(-1, 1)
    denormalizedDataset = np.ravel(scaler.inverse_transform(datReshaped))
    denormalizedDataset = denormalizedDataset.tolist()
    return denormalizedDataset

def findIndexOfAnomaly():
    anIndx = []
    for j in range(len(anomalyDenormalized)):
        idx = min(range(len(val)), key=lambda i: abs(val[i] - anomalyDenormalized[j]))
        anIndx.append(idx)
    return anIndx

def estimationPredictinForAnomaly(shape):
    y_pred = np.zeros(shape)
    y_pred = y_pred.tolist()

    for i in range(len(indexOfAnoamly)):
        index = indexOfAnoamly[i]
        y_pred[index] = 1.0
    return y_pred

def showPrecisionRatings():
    print("Recall:", precision_score(ytr, y_pred))
    print("Precision:", recall_score(ytr, y_pred))
    print("f1 score:", f1_score(ytr, y_pred))

def showPlotForDetectedAnomalies():
    plt.plot(arr, val, label='popyt')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(5))
    plt.plot(arr, tresholdDenormalized, label='treshold = ' + str(threshold), color='orange')
    plt.xticks(rotation=25)
    sns.scatterplot(x=indexOfAnoamly, y=anomalyDenormalized, s=12, label='anomalie', color='red')
    plt.xlabel('index czsowy')
    plt.ylabel('ilość')
    plt.title("Detekcja anomali dla pomytu godzinowego taxówek w NYC")
    plt.legend()
    plt.show()

def tresholdSelectionPlot():
    sns.displot(train_mae_los, bins=50, kde= True)
    plt.xlabel('ilość próbek')
    plt.ylabel('wartość znormalizowana')
    plt.title("Wykres częstości anomalii dla MAE wartości testowych")
    plt.show()

# parametry wejściowe dla sieci oraz przetworzonych danych
nUnits = 16
dropout = 0.2
batch_size = 32
epochs = 29
patience = 30
threshold = 1.2

# przygotowanie danych treningowych i testowych
X_train, X_test, y_train, y_test, scaler = prepare_and_load_data()

# trenowanie sieci na podstawie podzielonych danych
# model = trainEncoderDecoder(X_train, y_train, nUnits, dropout, batch_size, epochs)

#zapis modelu do pliku
# model.save(f'models/model_to_anomaly_deection_' + str(nUnits) + '_' + str(dropout) + '_' + str(epochs) + '.h5')

# odczyt przetrenowanego modelu z pliku
model = load_model('models/model_to_anomaly_deection.h5')
X_train_pred = model.predict(X_train)

#przygotowanie danych do prezentacji na wykresie
train, test = prepareDataToPresent()

# predykcja wartości
X_test_pred = model.predict(X_test)

#zmiana wymiarów tablic
X_train_pred_1D, X_train_1D, ytr = changeDimension(X_train_pred.shape[1], X_train_pred.shape[0])

#obliczenie mean absolut error na podstawie predykcji i danych treningowych w celu oszacoania anomalii
train_mae_los = np.mean(np.abs(X_train_pred_1D - X_train_1D), axis=1)

test_score_df, anomalies = anomalyDetectionBasedTreshold()

#ostateczne przygotowanie x i y do wyrysowania na wykresie
arr, val = xANDyPreparation(X_train_pred.shape[1]* X_train_pred.shape[0])

#denormalizacja danych
anomalyDenormalized = denormalizeDataSet(anomalies.loss)
tresholdDenormalized = denormalizeDataSet(test_score_df.threshold)

#znalezienie indexu poszczególnej anomali w zbiorze wartości predykcji
indexOfAnoamly = findIndexOfAnomaly()
y_pred = estimationPredictinForAnomaly(X_train_pred.shape[1]* X_train_pred.shape[0])

#wyliczenie Recall, Precision i F1 Scora
showPrecisionRatings()

#prezentowanie wyników końcowych
showPlotForDetectedAnomalies()

#wykres służący do doboru parametru treshold
# tresholdSelectionPlot()