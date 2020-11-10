import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Read Dataset
df = pd.read_csv("data/SouthGermanCredit.csv", sep=",")

# Get dummies
df = pd.get_dummies(df, drop_first=True)

# Normalize variables
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['credit_risk_good']),
                                                    df['credit_risk_good'],
                                                    test_size=0.3)
# Model
model = keras.Sequential()
model.add(layers.Dense(128, activation="relu", input_shape=(df.shape[1] - 1,)))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

# Train model
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.SGD(),
    metrics=["AUC"],
)

history = model.fit(X_train,
                    y_train,
                    batch_size=64,
                    epochs=500,
                    validation_split=0.2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=50)])

test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test AUC:", test_scores[1])

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model Performance')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
