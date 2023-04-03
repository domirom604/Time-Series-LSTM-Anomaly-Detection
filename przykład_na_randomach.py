import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# definicja sieci enkodera-dekodera
def autoencoder(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = tf.keras.layers.Dropout(0.2)(encoded)
    encoded = tf.keras.layers.LSTM(32, activation='relu')(encoded)
    encoded = tf.keras.layers.Dropout(0.2)(encoded)
    decoded = tf.keras.layers.RepeatVector(input_dim)(encoded)
    decoded = tf.keras.layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = tf.keras.layers.Dropout(0.2)(decoded)
    decoded = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = tf.keras.layers.Dropout(0.2)(decoded)
    decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim))(decoded)
    autoencoder = tf.keras.models.Model(inputs, decoded)
    return autoencoder

# funkcja trenująca sieć enkodera-dekodera i zwracająca próg anomalii
def trainAutoencoder(X_train):
    autoencoder_model = autoencoder(X_train.shape[1])
    autoencoder_model.compile(optimizer='adam', loss='mse')
    autoencoder_model.fit(X_train, X_train, epochs=50, batch_size=128, verbose=0)
    # obliczenie błędu rekonstrukcji na danych treningowych
    train_pred = autoencoder_model.predict(X_train)
    train_mae_loss = np.mean(np.abs(train_pred - X_train), axis=1)
    # wyznaczenie progu anomalii
    threshold = np.max(train_mae_loss)
    return threshold

# funkcja wykrywająca anomalie i obliczająca miarę F1
def detectAnomalies(X_test, y_test, threshold):
    autoencoder_model = autoencoder(X_test.shape[1])
    autoencoder_model.compile(optimizer='adam', loss='mse')
    autoencoder_model.fit(X_test, X_test, epochs=50, batch_size=128, verbose=0)
    # obliczenie błędu rekonstrukcji na danych testowych
    test_pred = autoencoder_model.predict(X_test)
    test_mae_loss = np.mean(np.abs(test_pred - X_test), axis=1)
    # wyznaczenie wykrytych anomalii na podstawie progu
    y_pred = np.zeros_like(test_mae_loss)
    y_pred[test_mae_loss > threshold] = 1
    # obliczenie miary F1
    f1 = f1_score(y_test, y_pred)
    return y_pred, f1

# przykładowe użycie
X = np.random.rand(1000, 10)
y = np.zeros(X.shape[0])
y[50:100] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
threshold = trainAutoencoder(X_train)
y_pred, f1 = detectAnomalies(X_test,y_test, threshold)
