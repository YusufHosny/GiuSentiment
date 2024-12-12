import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import F1Score, Precision, Recall
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

"""
Baseline code from "Neural Networks for Multi-Class Classification

Starts Here
"""

# Sequential model
model = Sequential(
    [
        tf.keras.Input(shape=(48, 48, 1)),
        tf.keras.layers.Flatten(),
        Dense(2500, activation='relu'),
        Dense(2500, activation='relu'),
        Dense(2000, activation='relu'),
        Dense(1500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(150, activation='relu'),
        Dense(7, activation='linear')
    ], name = "Baseline_model"
)

print(model.summary())


model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=['accuracy', Precision(), Recall(), F1Score()]
)

model.fit(
    train_ds,
    epochs=5
)

"""
Baseline code from "Neural Networks for Multi-Class Classification

Ends Here
"""


# confusion matrix
X_test = np.array([x for xbatch, _ in test_ds for x in xbatch])
y_test = np.array([y for _, ybatch in test_ds for y in ybatch])

y_test_pred = model.predict(X_test)

cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y_test_pred, axis=1))
cmd = ConfusionMatrixDisplay(cm, display_labels=['Angry', 'Disgusted', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'])

cmd.plot()
plt.show()