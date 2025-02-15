import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Bidirectional, LSTM, TimeDistributed, Reshape
from tensorflow.keras.models import Sequential

# Define model
model = Sequential()

# CNN for feature extraction
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 500, 3)))  # 3 channels for Δ, Δ²
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())  # Convert CNN output to 1D vector

# Reshape CNN output for BiLSTM
model.add(Reshape((62, 128)))  # 62 timesteps, 128 features

# BiLSTM for sequential feature learning
model.add(Bidirectional(LSTM(128, return_sequences=True)))

# Optional: TimeDistributed if frame-level predictions are needed
model.add(TimeDistributed(Dense(64, activation='relu')))

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  # num_classes = number of emotion categories

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
