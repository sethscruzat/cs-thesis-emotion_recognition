import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
import joblib
import cv2
import psutil
import tracemalloc
import os
import time
import assemblyai as aai
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Load Models
nb_model = joblib.load("./combined_model/models/best_model.pkl")# Load trained Naïve Bayes model
vectorizer = joblib.load("./combined_model/models/tfidf_vectorizer.pkl")

sterm_model = load_model("./combined_model/models/sterm.keras")
load_dotenv()

class_labels = {-1: "incorrect", 0: "negative", 1: "neutral", 2:"positive"}
class_labels_reversed = { "negative": 0, "neutral": 1, "positive": 2, "incorrect": -1}

# Function to measure memory usage
def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss

# ============================================================== SPEECH PREPROCESS =======================================================
def reduce_noise(y, sr):
    # Estimate noise profile from the first 0.5 seconds
    noise_sample = y[:int(sr * 0.5)]
    reduced_y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    return reduced_y

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_spec_normalized = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX) # Normalize spectrogram values to 0-255
    mel_spec_resized = cv2.resize(mel_spec_normalized, (128, 256),interpolation=cv2.INTER_AREA) # Resize to match CNN input

    spectogram = np.array(mel_spec_resized).reshape(-1, 128, 256, 1)
    spectogram = spectogram / 255.0
    
    return spectogram

# ============================================================== SPEECH MODEL =======================================================
def predict_audio(audio_input):
    tracemalloc.start()
    start_mem = get_memory_usage()
    start_time = time.time()

    sterm_probabilities = sterm_model.predict(audio_input)[0]
    
    end_time = time.time()
    end_mem = get_memory_usage()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Speech Model Prediction: {class_labels[np.argmax(sterm_probabilities)]}") #Print emotion label

    return np.argmax(sterm_probabilities), end_time - start_time, end_mem - start_mem, peak_mem

# ============================================================== TEXT PREPROCESS =======================================================
def speech_to_text(audio_file):
    aai.settings.api_key = os.getenv("AAI_API_KEY")

    # Transcribes audio file and extracts text
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed")
    
    print(f"Transcribed Text: {transcript.text}")
    return [transcript.text]

def preprocess_text(text):
    text = text.lower() # Lowercase text
    text = ''.join([char for char in text if char.isalpha() or char.isspace()]) # Remove punctuation, numbers, etc.
    return text

# ============================================================== TEXT MODEL =======================================================
def predict_text(text):
    tracemalloc.start()
    start_mem = get_memory_usage()
    start_time = time.time()

    processed_data = [preprocess_text(text) for text in text]
    # Convert text to feature vector
    text_vector = vectorizer.transform(processed_data)  # Convert to TF-IDF features
    # Get Naïve Bayes probabilities
    nb_probabilities = nb_model.predict_proba(text_vector)[0]

    end_time = time.time()
    end_mem = get_memory_usage()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Prediction using the loaded model
    prediction = nb_model.predict(text_vector)
    print(f"Text Model Prediction: {prediction}")

    return np.argmax(nb_probabilities), end_time - start_time, end_mem - start_mem, peak_mem

# ============================================================== FUSION MODEL =======================================================
def strict_fusion_predictions(audio_input, text):
    # Get predictions from both models
    speech_pred, cnn_time, cnn_mem, cnn_peak = predict_audio(audio_input)
    text_pred, nb_time, nb_mem, nb_peak = predict_text(text)

    # Strict agreement: both predictions must match and be correct
    if speech_pred == text_pred:
        return (
            class_labels[speech_pred],
            cnn_time, nb_time,
            cnn_mem, nb_mem,
            cnn_peak, nb_peak
            )
    else:
        return (
            "incorrect",
            cnn_time, nb_time,
            cnn_mem, nb_mem,
            cnn_peak, nb_peak
            )

# ============================================================== PREDICTION =======================================================

audio_file = "./combined_model/prediction_test/negative_test_6seconds.wav" # audio file
text = speech_to_text(audio_file)
audio_input = preprocess_audio(audio_file)

(
    predicted_emotion, 
    cnn_time, nb_time, 
    cnn_mem, nb_mem, 
    cnn_peak, nb_peak
) = strict_fusion_predictions(audio_input, text)

# ============================================================== RESULTS =======================================================
if predicted_emotion == "incorrect":
    print(f"Final Predicted Emotion: {predicted_emotion}")
else:
    print(f"Final Predicted Emotion: {class_labels[predicted_emotion]}")
print("\n===== TIME COMPLEXITY (Seconds) =====")
print(f"Speech Model: {cnn_time:.4f} sec")
print(f"Text Model: {nb_time:.4f} sec")

print("\n===== SPACE COMPLEXITY (Memory Usage in Bytes) =====")
print(f"Speech Model: {cnn_mem} bytes | Peak: {cnn_peak} bytes")
print(f"Text Model: {nb_mem} bytes | Peak: {nb_peak} bytes")