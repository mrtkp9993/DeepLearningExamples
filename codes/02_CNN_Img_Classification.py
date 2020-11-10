# I got an error on my system when I try to train this neural net
# `Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized`
# Solution: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode='nearest')

train_it = datagen.flow_from_directory('data/weather/train/',
                                       target_size=(128, 128),
                                       color_mode='rgb',
                                       class_mode='categorical',
                                       batch_size=16)
valid_it = datagen.flow_from_directory('data/weather/validation/',
                                       target_size=(128, 128),
                                       color_mode='rgb',
                                       class_mode='categorical',
                                       batch_size=16)

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4))
model.summary()

model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_it,
                    steps_per_epoch=len(train_it),
                    validation_data=valid_it,
                    validation_steps=len(valid_it),
                    epochs=50,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Performance')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
