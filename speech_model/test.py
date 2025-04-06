import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Bidirectional, LSTM, Reshape, BatchNormalization, GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.models import Sequential
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax

model = Sequential()

model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu'), input_shape=(1, 128, 256, 1)))

model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((3, 3))))

model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((3, 3))))

model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((3, 3))))
model.add(TimeDistributed(BatchNormalization()))

model.add(TimeDistributed(GlobalAveragePooling2D()))  # Keeps time steps intact

model.add(Bidirectional(LSTM(256, return_sequences=False)))

# Fully connected layers
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.45))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0005), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), metrics=['accuracy'])

model.summary()