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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
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
model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.003), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.25), metrics=['accuracy'])

model.summary()

df = pd.read_csv("./speech_model/label/all_labels_six.csv")

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
    return cv2.resize(image, (256, 128), interpolation=cv2.INTER_AREA)

load_dataset("./speech_model/all_spectrograms/six_seconds")

X_resized = np.array([resize_spectrogram(img) for img in X])  # X contains spectrograms

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)  # Convert labels to numbers

# Indices for each class
positive_indices = np.where(y_encoded == 2)[0]  # Positive
neutral_indices = np.where(y_encoded == 1)[0]   # Neutral
negative_indices = np.where(y_encoded == 0)[0]  # Negative

# Downsample negative samples to match positive class size
n_samples = len(neutral_indices)
balanced_negative_indices = np.random.choice(negative_indices, n_samples, replace=False)
balanced_positive_indices = np.random.choice(positive_indices, n_samples, replace=False)

# Combine indices for a balanced dataset
balanced_indices = np.concatenate([balanced_positive_indices, neutral_indices, balanced_negative_indices])

# Shuffle the indices
np.random.shuffle(balanced_indices)

# Create the balanced dataset
X_balanced = X_resized[balanced_indices]
y_balanced = y_encoded[balanced_indices]

y_onehot = to_categorical(y_balanced)  # Convert to one-hot encoding

X_balanced = np.array(X_balanced).reshape(-1, 128, 256, 1)  # Reshape for CNN (assuming grayscale images)
X_balanced = X_balanced / 255.0  # Normalize pixel values (for image-based spectrograms)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_onehot, test_size=0.2, random_state=42, shuffle=True)

X_train = np.expand_dims(X_train, axis=1)  # Adds a time dimension
X_test = np.expand_dims(X_test, axis=1)  # Adds a time dimension

early_stopping = EarlyStopping(monitor='val_accuracy', patience=14, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1, callbacks=[early_stopping])

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

    plt.savefig("cnn_confusion_matrix.png")  # Save figure
    plt.close()

    # 5. Print classification report
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print(confusion_matrix(y_true, y_pred))

model_testing()

model.save("cnn_six_seconds.keras")