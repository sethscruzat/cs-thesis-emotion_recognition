import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, Bidirectional, GRU, Reshape, BatchNormalization
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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax

model = Sequential()

# Lightweight CNN feature extraction, 3 layers
model.add(SeparableConv2D(64, (3, 3), activation='relu', input_shape=(128, 256, 1)))

model.add(SeparableConv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3)))

model.add(SeparableConv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3)))
model.add(Dropout(0.25))

model.add(SeparableConv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3)))

# # Reshape CNN output for BiGRU
# model.add(Reshape((-1, 128)))

# # BiGRU for sequential learning
# model.add(Bidirectional(GRU(256, return_sequences=True)))
# model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(Dropout(0.50))

model.add(Flatten())  # Convert CNN output to 1D vector

# Fully connected layers
model.add(Dense(1024, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

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
    return cv2.resize(image, (256, 128), interpolation=cv2.INTER_AREA)

load_dataset("./speech_model/all_spectrograms/")

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

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_onehot, test_size=0.2, random_state=13, shuffle=True)

# class_labels = np.unique(np.argmax(y_train, axis=1)) 
# class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=np.argmax(y_train, axis=1))
# class_weights_dict = dict(enumerate(class_weights))
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1, class_weight=class_weights_dict)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

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