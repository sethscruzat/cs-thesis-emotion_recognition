import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, Bidirectional, GRU, Reshape
from tensorflow.keras.models import Sequential
import tensorflow_model_optimization as tfmot

# Pruning wrapper
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

model = Sequential()

# Lightweight CNN feature extraction, 3 layers
model.add(SeparableConv2D(16, (3, 3), activation='relu', input_shape=(128, 500, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(SeparableConv2D(32, (3, 3), activation='relu'))  
model.add(MaxPooling2D((2, 2)))

model.add(SeparableConv2D(64, (3, 3), activation='relu'))  
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())  # Convert CNN output to 1D vector

# Reshape CNN output for BiGRU
model.add(Reshape((62, 64)))  # 62 timesteps, 64 features

# BiGRU for sequential learning
model.add(prune_low_magnitude(Bidirectional(GRU(64, return_sequences=True))))

# Fully connected layers
model.add(prune_low_magnitude(Dense(64, activation='relu')))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  

# Apply quantization for model compression
# quantize_model = tfmot.quantization.keras.quantize_model(model)
# quantize_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# quantize_model.summary()

# pruning to filter out weights with less value
model = prune_low_magnitude(model)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # NOTE: change metrics later

model.summary()


# TODO IMPLEMENT TRAINING AND TESTING