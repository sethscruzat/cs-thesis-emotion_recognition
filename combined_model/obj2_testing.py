import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
import joblib
import cv2
import psutil
import tracemalloc  # Memory tracking
import os
import time
from sklearn.preprocessing import StandardScaler
import assemblyai as aai
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import gc

class_labels = {2:"Positive", 1: "Neutral", 0: "Negative"}

# Load models
cnn_model = load_model("./combined_model/results/OBJ_2/six_seconds/cnn_six_seconds.keras")  # Load trained CNN model
benchmark_model = load_model("./combined_model/results/OBJ_2/six_seconds/benchmark_six_seconds.keras")  # Load trained benchmark model

# Function to measure memory usage
def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss  # Memory in bytes

# ============================================================== PREPROCESSING =======================================================
def reduce_noise(y, sr):
    """Apply noise reduction using noisereduce."""
    # Estimate noise profile from the first 0.5 seconds
    noise_sample = y[:int(sr * 0.5)]
    reduced_y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    return reduced_y

def preprocess_audio_cnn(audio_file, target_size=(128, 256)):
    y, sr = librosa.load(audio_file, sr=22050)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize spectrogram values to 0-255 (image-like format)
    mel_spec_normalized = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX)

    # Resize to match CNN input
    mel_spec_resized = cv2.resize(mel_spec_normalized, target_size)

    spectogram = np.array(mel_spec_resized).reshape(-1, 128, 256, 1)
    spectogram = spectogram / 255.0

    audio_input = np.expand_dims(spectogram, axis=1)

    return audio_input

def preprocess_audio_benchmark(audio_file, target_size=(128, 256)):
    y, sr = librosa.load(audio_file, sr=22050)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize spectrogram values to 0-255 (image-like format)
    mel_spec_normalized = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX)

    # Resize to match CNN input
    mel_spec_resized = cv2.resize(mel_spec_normalized, target_size)

    spectogram = np.array(mel_spec_resized).reshape(-1, 128, 256, 1)
    spectogram = spectogram / 255.0

    return spectogram

# ============================================================== SPEECH MODEL =======================================================
def predict_audio_cnn(audio_input):
    gc.collect()
    tracemalloc.stop()            # Stop any previous tracing
    tracemalloc.clear_traces()    # Clear memory snapshot history 

    tracemalloc.start()
    start_mem = get_memory_usage()
    start_time = time.time()

    # Get CNN probabilities
    cnn_probabilities = cnn_model.predict(audio_input)[0]
    predicted_class = np.argmax(cnn_probabilities)  # Get index of highest probability

    end_time = time.time()
    end_mem = get_memory_usage()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predicted_emotion = class_labels[predicted_class]  # Convert index to label
    print(f"CNN Model Prediction: {predicted_emotion}")
    print("Confidence Scores:", cnn_probabilities)

    return end_time - start_time, end_mem - start_mem, peak_mem

def predict_audio_benchmark(audio_input):
    tracemalloc.start()
    start_mem = get_memory_usage()
    start_time = time.time()

    # Get CNN probabilities
    benchmark_probabilities = benchmark_model.predict(audio_input)[0]
    predicted_class = np.argmax(benchmark_probabilities)  # Get index of highest probability

    end_time = time.time()
    end_mem = get_memory_usage()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predicted_emotion = class_labels[predicted_class]  # Convert index to label
    print(f"Benchmark Model Prediction: {predicted_emotion}")
    print("Confidence Scores:", benchmark_probabilities)

    return end_time - start_time, end_mem - start_mem, peak_mem

# ============================================================== PROCESS START =======================================================

audio_file = "./combined_model/test_prediction.wav"  # audio file
audio_input_benchmark = preprocess_audio_benchmark(audio_file)
audio_input_cnn = preprocess_audio_cnn(audio_file)

gc.collect()
tracemalloc.stop()
tracemalloc.clear_traces()
cnn_time, cnn_mem, cnn_peak = predict_audio_cnn(audio_input_cnn)

gc.collect()
tracemalloc.stop()
tracemalloc.clear_traces()
benchmark_time, benchmark_mem, benchmark_peak = predict_audio_benchmark(audio_input_benchmark)

# ============================================================== RESULTS =======================================================
print("\n===== TIME COMPLEXITY (Seconds) =====")
print(f"CNN Model: {cnn_time:.4f} sec")
print(f"Benchmark Model: {benchmark_time:.4f} sec")

print("\n===== SPACE COMPLEXITY (Memory Usage in Bytes) =====")
print(f"CNN Model: {cnn_mem} bytes | Peak: {cnn_peak} bytes")
print(f"Benchmark Model: {benchmark_mem} bytes | Peak: {benchmark_peak} bytes")