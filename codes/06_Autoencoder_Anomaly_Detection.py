import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat
from sklearn.preprocessing import Normalizer
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

# Read data
data = loadmat("data/breastw.mat")

X = data['X']
y = data['y']
X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Normalize data
scaler = Normalizer()
X_scaled = pd.DataFrame(scaler.fit_transform(X))

df = pd.concat([X_scaled, y], ignore_index=True, axis=1)
df.columns = [f'X{i}' for i in range(1, X.shape[1] + 1)] + ['isOutlier']

# Split normal data and outliers
df_normal = df[df.isOutlier == 0]
df_normal.columns = [f'X{i}' for i in range(1, X.shape[1] + 1)] + ['isOutlier']

df_outlier = df[df.isOutlier == 1]
df_outlier.columns = [f'X{i}' for i in range(1, X.shape[1] + 1)] + ['isOutlier']

df_normal.drop(["isOutlier"], axis=1, inplace=True)
df_outlier.drop(['isOutlier'], axis=1, inplace=True)

# Split normal data as train and validation
train_size = int(len(df_normal) * 0.67)
valid_size = len(df_normal) - train_size

train_normal = df_normal.iloc[[i for i in range(train_size)], :]
valid_normal = df_normal.iloc[[i for i in range(train_size, len(df_normal))], :]


# Autoencoder model
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(9, activation='elu'),
            layers.Dense(6, activation='elu'),
            layers.Dense(3, activation='elu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(3, activation='elu'),
            layers.Dense(6, activation='elu'),
            layers.Dense(9, activation='elu')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder()

autoencoder.compile(optimizer=tf.keras.optimizers.SGD(),
                    loss=losses.mean_absolute_error,
                    metrics=['mean_absolute_error'])

# Fit on normal train data
history = autoencoder.fit(train_normal, train_normal,
                          epochs=100,
                          batch_size=100,
                          validation_data=(valid_normal, valid_normal),
                          callbacks=[EarlyStopping(patience=10)])

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model Performance')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Calculate reconstruction errors for normal and outliers
train_normal_pred = autoencoder.predict(train_normal)
train_mae_loss = np.mean(np.abs(train_normal_pred - train_normal), axis=1)
outlier_pred = autoencoder.predict(df_outlier)
outlier_mae_loss = np.mean(np.abs(outlier_pred - df_outlier), axis=1)

plt.hist(train_mae_loss)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

plt.hist(outlier_mae_loss)
plt.xlabel("Outlier MAE loss")
plt.ylabel("No of samples")
plt.show()

# Calculate MAE threshold for anomalies
# I'll choose threshold as median loss
threshold = np.median(train_mae_loss)
print("Reconstruction error threshold: ", threshold)
false_alarms = train_mae_loss > threshold
anomalies = outlier_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

# F1 score for classification
f1_score = np.sum(anomalies) / (np.sum(anomalies) + 1 / 2 * (np.sum(false_alarms) + anomalies.size - np.sum(anomalies)))
print(f1_score)
