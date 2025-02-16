import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Bidirectional, LSTM, TimeDistributed, Reshape
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

# Define model
model = Sequential()

# CNN for feature extraction
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 500, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())  # Convert CNN output to 1D vector

# Reshape CNN output for BiLSTM
model.add(Reshape((-1, 128)))

# BiLSTM for sequential feature learning
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(tf.keras.layers.GlobalAveragePooling1D())

# # Optional: TimeDistributed if frame-level predictions are needed
# model.add(TimeDistributed(Dense(64, activation='relu')))

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # num_classes = number of emotion categories

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

df = pd.read_csv("./speech_model/labels/all_labels.csv")

X = [] # spectrograms
Y = [] # labels

def load_dataset(spectrogram_folder):
    for _, row in df.iterrows():
        spec_path = os.path.join(spectrogram_folder, row["filename"])
        spectrogram = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
        
        X.append(spectrogram)
        Y.append(row["emotion"])  # appends label

# Example function to resize spectrograms
def resize_spectrogram(image):
    return cv2.resize(image, (500, 128), interpolation=cv2.INTER_AREA)


load_dataset("./speech_model/all_spectrograms/")

X_resized = np.array([resize_spectrogram(img) for img in X])  # X contains spectrograms

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)  # Convert labels to numbers
y_onehot = to_categorical(y_encoded)  # Convert to one-hot encoding

X_resized = np.array(X_resized).reshape(-1, 128, 500, 1)  # Reshape for CNN (assuming grayscale images)
X_resized = X_resized / 255.0  # Normalize pixel values (for image-based spectrograms)

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Stop if no improvement for 5 epochs
    restore_best_weights=True  # Restore the best model weights
)

X_train, X_test, y_train, y_test = train_test_split(X_resized, y_onehot, test_size=0.2, random_state=42, shuffle=True)

print("Train size:", len(X_train), len(y_train))
print("Test size:", len(X_test), len(y_test))

# early stopping
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[early_stopping])

# # Train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

def model_testing():
    # 1. Get predictions (probabilities)
    y_pred_probs = model.predict(X_test)  # Predict on test data

    # 2. Convert to class labels
    y_pred = np.argmax(y_pred_probs, axis=1)  # Get index of max probability
    y_true = np.argmax(y_test, axis=1)  # Convert one-hot labels back to integers

    # 3. Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # 4. Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # 5. Print classification report
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

model_testing()