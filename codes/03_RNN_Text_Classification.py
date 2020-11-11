import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

with open("data/clinc150_full.json", "r") as read_file:
    datajson = json.load(read_file)

df_train = pd.DataFrame(datajson['train'], columns=['Text', 'Class'])
df_val = pd.DataFrame(datajson['val'], columns=['Text', 'Class'])

maxlen = 100
training_samples = df_train.shape[0]
validation_samples = df_val.shape[0]
max_words = 1000
num_classes = len(df_train.Class.unique())

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_train.Text)

sequences_train = tokenizer.texts_to_sequences(df_train.Text)
sequences_valid = tokenizer.texts_to_sequences(df_val.Text)

x_train = pad_sequences(sequences_train, maxlen=maxlen)
x_valid = pad_sequences(sequences_valid, maxlen=maxlen)

y_train = df_train.Class.values
y_valid = df_val.Class.values

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_y_train = encoder.transform(y_train)
encoded_y_valid = encoder.transform(y_valid)

dummy_y_train = to_categorical(encoded_y_train)
dummy_y_valid = to_categorical(encoded_y_valid)

model = Sequential()
model.add(layers.Embedding(max_words, output_dim=32))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(num_classes, activation="softmax"))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train,
                    dummy_y_train,
                    batch_size=128,
                    epochs=100,
                    validation_data=(x_valid, dummy_y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Performance')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
