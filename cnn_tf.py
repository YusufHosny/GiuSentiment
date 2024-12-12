import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, ReLU, Dropout, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def load_data():
    
    startpath = os.path.join('dataset', 'train')
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        startpath,
        validation_split=0.2,
        subset="both",
        seed=123,
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(48, 48),
        batch_size=25)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    startpath = os.path.join('dataset', 'test')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        startpath,
        seed=123,
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(48, 48),
        batch_size=25)
        
    return train_ds, val_ds, test_ds


# load dataset
train_ds, val_ds, test_ds = load_data()

# Sequential model
model = Sequential(
    [
        tf.keras.Input(shape=(48, 48, 1)),
        Conv2D(64, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(64, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPool2D((2, 2), strides=2),
        Conv2D(64, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(128, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPool2D((2, 2), strides=2),
        Conv2D(128, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(256, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPool2D((2, 2), strides=2),
        Conv2D(256, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(512, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPool2D((2, 2), strides=2),
        Flatten(),
        Dense(4096),
        Dropout(0.2),
        ReLU(),
        Dense(4096),
        Dropout(0.2),
        ReLU(),
        Dense(7)
    ], name = "VGG_model"
)

print(model.summary())


model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=['accuracy', tf.keras.metrics.F1Score()]
)

model.fit(
    train_ds,
    epochs=5
)


# confusion matrix
X_test = np.array([x for xbatch, _ in test_ds for x in xbatch])
y_test = np.array([y for _, ybatch in test_ds for y in ybatch])

y_test_pred = model.predict(X_test)

cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y_test_pred, axis=1))
cmd = ConfusionMatrixDisplay(cm, display_labels=['Angry', 'Disgusted', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'])

cmd.plot()
plt.show()