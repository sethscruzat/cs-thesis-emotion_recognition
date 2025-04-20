import tensorflow as tf
import cv2
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Bidirectional, LSTM, Reshape, BatchNormalization, GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ============================================================== MODEL START ===========================================================
"""
    Filters = 64, 128, 128, 256, 256
    droupout layer = 512
    kernel size = 3 x 3
    pooling size = 3 x 3
    input shape = 128 x 256 x 1
    channels = 1 (grayscale, 3 = rgb)
    number of dense neurons = 3 (3 classes)
"""

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

# ============================================================== PROCESSING ===============================================================
df = pd.read_csv("./speech_model/final/all_labels_six.csv") # load labels

X = [] # spectrograms
Y = [] # labels

def load_dataset(spectrogram_folder):
    for _, row in df.iterrows():
        spec_path = os.path.join(spectrogram_folder, row["filename"])
        spectrogram = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
        
        X.append(spectrogram)
        Y.append(row["emotion"])  # appends label

def resize_spectrogram(image):
    return cv2.resize(image, (256, 128), interpolation=cv2.INTER_AREA)

load_dataset("./speech_model/final/six_seconds")

X_resized = np.array([resize_spectrogram(img) for img in X])  # resizes all spectrograms in array X

label_encoder = LabelEncoder() # encodes emotion classes into numbers {0: negative, 1: neutral, 2: positive}
y_encoded = label_encoder.fit_transform(Y)  # Convert labels in Y array to numbers

# Indices for each class
positive_indices = np.where(y_encoded == 2)[0] 
neutral_indices = np.where(y_encoded == 1)[0]  
negative_indices = np.where(y_encoded == 0)[0] 

# Downsample negative samples to match positive class size. this is for balancing the dataset
n_samples = len(neutral_indices)
balanced_negative_indices = np.random.choice(negative_indices, n_samples, replace=False)
balanced_positive_indices = np.random.choice(positive_indices, n_samples, replace=False)

balanced_indices = np.concatenate([balanced_positive_indices, neutral_indices, balanced_negative_indices]) # Combining indices
np.random.shuffle(balanced_indices) # Shuffle the indices

# Creating balanced dataset
X_balanced = X_resized[balanced_indices]
y_balanced = y_encoded[balanced_indices]

y_onehot = to_categorical(y_balanced)  # Convert to one-hot encoding

X_balanced = np.array(X_balanced).reshape(-1, 128, 256, 1)  # Reshape for CNN
X_balanced = X_balanced / 255.0  # Normalizing pixel values

# ============================================================== TRAINING =================================================================
# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_onehot, test_size=0.25, random_state=42, shuffle=True)

# Adds a time dimension
X_train = np.expand_dims(X_train, axis=1)  
X_test = np.expand_dims(X_test, axis=1)

# early stopping parameters. stops model training when improvement hasn't been made for 12 epochs. returns best model
early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=1, callbacks=[early_stopping])

# ============================================================== TESTING =================================================================
def model_testing():
    # Make predictions (probabilities)
    y_pred_probs = model.predict(X_test)  # Predict on test data

    # Convert to class labels
    y_pred = np.argmax(y_pred_probs, axis=1)  # Get index of max probability
    y_true = np.argmax(y_test, axis=1)  # Convert one-hot labels back to integers

    # Plotting confusion matrix
    cm = confusion_matrix(y_true, y_pred)
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
model.save("cnn_six_seconds.keras") # save model