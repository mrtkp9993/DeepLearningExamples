import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Read dataset
df = pd.read_csv('data/SeoulBikeData.csv',
                 parse_dates=['Date'],
                 encoding='cp1250')

# Date features
df['Date_Month'] = df['Date'].apply(lambda x: x.month)
df['Date_Day'] = df['Date'].apply(lambda x: x.day)
df['Date_Weekday'] = df['Date'].apply(lambda x: x.weekday())
df.drop(columns=["Date"], inplace=True)

# Get dummies
df = pd.get_dummies(df, drop_first=True,
                    columns=['Seasons', 'Holiday', 'Functioning Day'])

# Normalize variables
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Rented Bike Count']),
                                                    df['Rented Bike Count'],
                                                    test_size=0.3)

# Model
model = keras.Sequential()
model.add(layers.Dense(128, activation="relu", input_shape=(df.shape[1] - 1,)))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="relu"))

model.summary()

# Train model
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.SGD(),
    metrics=["mean_squared_error"],
)

history = model.fit(X_train,
                    y_train,
                    batch_size=64,
                    epochs=400,
                    validation_split=0.2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=20)])

test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test MSE:", test_scores[1])

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Model Performance')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
