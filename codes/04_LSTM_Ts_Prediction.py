import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

data = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")

lookback = 24

scaler = MinMaxScaler()
df = scaler.fit_transform(data.traffic_volume.values.reshape((-1, 1)))
train_size = int(len(df) * 0.67)
valid_size = len(df) - train_size
train, valid = df[0:train_size, :], df[train_size:len(df), :]

train_data_gen = TimeseriesGenerator(train, train,
                                     length=lookback, sampling_rate=1, stride=1,
                                     batch_size=8)
valid_data_gen = TimeseriesGenerator(valid, valid,
                                     length=lookback, sampling_rate=1, stride=1,
                                     batch_size=1)

model = Sequential()
# GRU can be replaced by LSTM layer
model.add(layers.GRU(4, input_shape=(lookback, 1), return_sequences=True))
model.add(layers.Dense(1))
model.summary()

model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mape'])

# https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio
tf.config.experimental_run_functions_eagerly(True)
history = model.fit(train_data_gen,
                    validation_data=valid_data_gen,
                    epochs=100,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)])

plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'])
plt.title('Model Performance')
plt.ylabel('MAPE')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
