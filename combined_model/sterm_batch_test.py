import numpy as np
import tensorflow as tf
import pandas as pd
from pydub import AudioSegment
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import cv2
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import assemblyai as aai
from tensorflow.keras.models import load_model

# Load Models
nb_model = joblib.load("./combined_model/models/best_model.pkl")# Load trained Naïve Bayes model
vectorizer = joblib.load("./combined_model/models/tfidf_vectorizer.pkl")

sterm_model = load_model("./combined_model/models/sterm.keras")

audio_folder_path = "./speech_model/audio_2/wav"
csv_folder_path = "./speech_model/labels_2"

output_dir = "./combined_model/validation_test/extracted_audio"

sterm_weight = 0.36
sterm_nb_weight = 0.56

total_sterm = sterm_weight + sterm_nb_weight
sterm_weight /= total_sterm
sterm_nb_weight /= total_sterm

class_labels = {0: "negative", 1: "neutral", 2:"positive", -1: "incorrect"}
class_labels_reversed = { "negative": 0, "neutral": 1, "positive": 2}

def clean_output_dir():
    for file in os.listdir(output_dir):
        if file.endswith(".wav"):
            os.remove(os.path.join(output_dir, file))


def split_audio(long_audio_path, df):
    audio = AudioSegment.from_wav(long_audio_path)
    file_paths, true_labels = [], []

    for index, row in df.iterrows():
        start_time = row["start_time"] * 1000  # Convert to ms
        end_time = row["end_time"] * 1000
        emotion = row["emotion"]  # Integer class label

        # Extract segment
        segment = audio[start_time:end_time]
        segment_filename = f"{output_dir}/segment_{index}.wav"
        segment.export(segment_filename, format="wav")

        # Store file path & label
        file_paths.append(segment_filename)
        true_labels.append(int(class_labels_reversed[emotion]))

    return file_paths, true_labels


# SPEECH MODEL
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
    
    return spectogram


def predict_audio(audio_file):
    spectogram = audio_to_spectogram(audio_file)
    sterm_probabilities = sterm_model.predict(spectogram)[0]

    return np.argmax(sterm_probabilities)


# TEXT MODEL
def speech_to_text(audio_file):
    aai.settings.api_key = "c45ab6cc228640d58dfe8b8b43b712df"

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed")
    
    return [transcript.text]

def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove punctuation, numbers, etc.
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    return text

def predict_text(audio_file):
    transcript = speech_to_text(audio_file)
    processed_data = [preprocess_text(text) for text in transcript]
    # Convert text to feature vector
    text_vector = vectorizer.transform(processed_data)  # Convert to TF-IDF features

    # Get Naïve Bayes probabilities
    nb_probabilities = nb_model.predict_proba(text_vector)[0]  # (num_classes,)

    return np.argmax(nb_probabilities)

def strict_fusion_predictions(audio_files, y_true):
    y_pred = []
    
    for idx, file in enumerate(audio_files):
        speech_pred = predict_audio(file)
        text_pred= predict_text(file)
        true_label = y_true[idx]
        
        # Strict agreement: both predictions must match and be correct
        if speech_pred == text_pred == true_label:
            y_pred.append(true_label)  # Correct prediction
        else:
            y_pred.append(-1) 
    
    return y_pred

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
            y_pred = strict_fusion_predictions(audio_files, y_true)

            # Collect for global evaluation
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            clean_output_dir()

    return all_y_true, all_y_pred

# Run batch processing
all_y_true, all_y_pred = batch_process_all_files(audio_folder_path, csv_folder_path)
all_labels = [0, 1, 2, -1]
# Compute confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.savefig("sterm_confusion_matrix.png")  # Save figure
plt.close()

# Print classification report
report = classification_report(all_y_true, all_y_pred, labels=[0, 1, 2, -1], target_names=class_labels.values())
print(report)