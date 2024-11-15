import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from PIL import Image
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
        label_mode='int',
        color_mode='grayscale',
        image_size=(48, 48),
        batch_size=25)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
    return train_ds, val_ds


"""
Baseline code from "Neural Networks for Multi-Class Classification

Starts Here
"""

# load dataset
train_ds, val_ds = load_data()

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
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    epochs=5
)

"""
Baseline code from "Neural Networks for Multi-Class Classification

Ends Here
"""

# inference on 1 batch
for X, y in train_ds.take(1):
    pred = model.predict(X)
    print(pred, y)