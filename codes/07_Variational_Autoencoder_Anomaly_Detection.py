import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat
from sklearn.preprocessing import Normalizer
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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


# Sampling class
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Autoencoder model
latent_dim = 2

encoder_inputs = Input(shape=(9,))
x = layers.Dense(9, activation='elu')(encoder_inputs)
x = layers.Dense(6, activation='elu')(x)
x = layers.Dense(3, activation='elu')(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = Input(shape=(latent_dim,))
x = layers.Dense(3, activation="elu")(latent_inputs)
x = layers.Dense(6, activation="elu")(x)
x = layers.Dense(9, activation="elu")(x)
decoder_outputs = x
decoder = Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 9 * 9
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


vae = VAE(encoder, decoder)
vae.compile(optimizer=Adam(),
            metrics=['mean_absolute_error'],
            loss=losses.mean_absolute_error)
history = vae.fit(train_normal,
                  epochs=200,
                  batch_size=128,
                  validation_data=(valid_normal, valid_normal)
                  ).history

# Calculate reconstruction errors for normal and outliers
train_normal_pred = vae.predict(train_normal)
train_mae_loss = np.mean(np.abs(train_normal_pred - train_normal), axis=1)
outlier_pred = vae.predict(df_outlier)
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
threshold = np.mean(train_mae_loss)
print("Reconstruction error threshold: ", threshold)
false_alarms = train_mae_loss > threshold
anomalies = outlier_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

# F1 score for classification
f1_score = np.sum(anomalies) / (np.sum(anomalies) + 1 / 2 * (np.sum(false_alarms) + anomalies.size - np.sum(anomalies)))
print(f1_score)
