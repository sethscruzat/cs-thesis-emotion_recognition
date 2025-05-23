import numpy as np
import tensorflow as tf
import pandas as pd
import librosa
import joblib
import cv2
import psutil
import tracemalloc
import os
import time
import seaborn as sns
import assemblyai as aai
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load Models
cnn_model = load_model("./combined_model/models/cnn_ver2.keras")  # Load trained CNN model
nb_model = joblib.load("./combined_model/models/best_model.pkl")# Load trained Naïve Bayes model
vectorizer = joblib.load("./combined_model/models/tfidf_vectorizer.pkl")

audio_folder_path = "./speech_model/audio_2/wav"
csv_folder_path = "./speech_model/labels_2"

output_dir = "./combined_model/validation_test/extracted_audio"
load_dotenv()

# Weight definition
cnn_weight = 0.50 
nb_weight = 0.86 

# Normalize weights
total_weight = cnn_weight + nb_weight
cnn_weight /= total_weight
nb_weight /= total_weight

class_labels = {0: "negative", 1: "neutral", 2:"positive" }
class_labels_reversed = { "negative": 0, "neutral": 1, "positive": 2}

# cleans up extracted_audio folder after each iteration
def clean_output_dir():
    for file in os.listdir(output_dir):
        if file.endswith(".wav"):
            os.remove(os.path.join(output_dir, file))

# ============================================================== SPEECH PREPROCESS =======================================================
def split_audio(long_audio_path, df):
    audio = AudioSegment.from_wav(long_audio_path)
    file_paths, true_labels = [], []

    for index, row in df.iterrows():
        start_time = row["start_time"] * 1000  # Convert to ms
        end_time = row["end_time"] * 1000
        emotion = row["emotion"]

        # Extract segment
        segment = audio[start_time:end_time]
        segment_filename = f"{output_dir}/segment_{index}.wav"
        segment.export(segment_filename, format="wav")

        # Store file path & label
        file_paths.append(segment_filename)
        true_labels.append(int(class_labels_reversed[emotion]))

    return file_paths, true_labels

def reduce_noise(y, sr):
    # Estimate noise profile from the first 0.5 seconds
    noise_sample = y[:int(sr * 0.5)]
    reduced_y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    return reduced_y

def audio_to_spectogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_spec_normalized = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX)
    mel_spec_resized = cv2.resize(mel_spec_normalized, (128, 256))

    spectogram = np.array(mel_spec_resized).reshape(-1, 128, 256, 1)
    spectogram = spectogram / 255.0
    
    return np.expand_dims(spectogram, axis=1) # Reshape for CNN input

# ============================================================== SPEECH MODEL =======================================================
def predict_audio(audio_file):
    spectogram = audio_to_spectogram(audio_file)
    cnn_probabilities = cnn_model.predict(spectogram)[0]

    return np.argmax(cnn_probabilities), cnn_probabilities


# ============================================================== TEXT PREPROCESS =======================================================
def speech_to_text(audio_file):
    aai.settings.api_key = os.getenv("AAI_API_KEY")

    # Transcribes audio file and extracts text
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed")
    return [transcript.text]

def preprocess_text(text):
    text = text.lower() # Lowercase text
    text = ''.join([char for char in text if char.isalpha() or char.isspace()]) # Remove punctuation, numbers, etc.
    return text

# ============================================================== TEXT MODEL =======================================================
def predict_text(audio_file):
    transcript = speech_to_text(audio_file)

    processed_data = [preprocess_text(text) for text in transcript]
    text_vector = vectorizer.transform(processed_data)

    nb_probabilities = nb_model.predict_proba(text_vector)[0]
    return np.argmax(nb_probabilities), nb_probabilities

# ============================================================== FUSION MODEL =======================================================# Fusion Function (Weighted Confidence Score)
def fusion_prediction(speech_pred, speech_probs, text_pred, text_probs):
    final_scores = (cnn_weight * speech_probs) + (nb_weight * text_probs)

    return np.argmax(final_scores)

def batch_fusion_predictions(audio_files, y_true):
    y_pred = []
    
    for file in audio_files:
        speech_pred, speech_probs = predict_audio(file)
        text_pred, text_probs = predict_text(file)
        fused_pred = fusion_prediction(speech_pred, speech_probs, text_pred, text_probs)
        y_pred.append(fused_pred)

    return y_pred

# ============================================================== BATCH PREDICTION =======================================================
def batch_process_all_files(audio_folder_path, csv_folder_path):
    all_y_true = []
    all_y_pred = []

    for filename in os.listdir(audio_folder_path):
        if filename.endswith(".wav"):
            print(f"processing: {filename}")
            base_name = os.path.splitext(filename)[0]
            audio_path = os.path.join(audio_folder_path, filename)
            csv_path = os.path.join(csv_folder_path, f"{base_name}.csv")

            if not os.path.exists(csv_path):
                print(f"CSV not found for {filename}, skipping...")
                continue

            # Load CSV
            df = pd.read_csv(csv_path)

            # Segment and get true labels
            audio_files, y_true = split_audio(audio_path, df)

            # Get predictions
            y_pred = batch_fusion_predictions(audio_files, y_true)

            # Collect for global evaluation
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            clean_output_dir()

    return all_y_true, all_y_pred

# Run batch processing
all_y_true, all_y_pred = batch_process_all_files(audio_folder_path, csv_folder_path)

# ============================================================== RESULTS ===========================================================
cm = confusion_matrix(all_y_true, all_y_pred)
# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.savefig("batchtest_confusion_matrix.png")  # Save figure
plt.close()

# Print classification report
report = classification_report(all_y_true, all_y_pred, target_names=class_labels.values())
print(report)