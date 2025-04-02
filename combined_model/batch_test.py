import numpy as np
import tensorflow as tf
import pandas as pd
from pydub import AudioSegment
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import assemblyai as aai
from tensorflow.keras.models import load_model

# Load Models
cnn_model = load_model("./combined_model/models/cnn_six_seconds.keras")  # Load trained CNN model
nb_model = joblib.load("./combined_model/models/best_model.pkl")# Load trained Naïve Bayes model
vectorizer = joblib.load("./combined_model/models/tfidf_vectorizer.pkl")

# batch test
csv_path = "./combined_model/validation_test/Ses03F_script01_3.csv"
df = pd.read_csv(csv_path)

output_dir = "./combined_model/validation_test/extracted_audio"

# Define Weights (Based on Validation Accuracy)
cnn_weight = 0.52 
nb_weight = 0.86 

# Normalize weights so they sum to 1
total_weight = cnn_weight + nb_weight
cnn_weight /= total_weight
nb_weight /= total_weight

class_labels = {2:"positive", 1: "neutral", 0: "negative"}
class_labels_reversed = {"positive": 2, "neutral": 1, "negative": 0}

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
    """Apply noise reduction using noisereduce."""
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
    
    # Reshape for CNN input
    return np.expand_dims(spectogram, axis=1)  # Shape: (1, 128, T, 1)


def predict_audio(audio_file):
    spectogram = audio_to_spectogram(audio_file)
    cnn_probabilities = cnn_model.predict(spectogram)[0]

    return np.argmax(cnn_probabilities), cnn_probabilities


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

    return np.argmax(nb_probabilities), nb_probabilities


# Fusion Function (Weighted Confidence Score)
def fusion_prediction(speech_pred, speech_probs, text_pred, text_probs):
    # Compute final weighted score
    final_scores = (cnn_weight * speech_probs) + (nb_weight * text_probs)

    return np.argmax(final_scores)

def batch_fusion_predictions(audio_files, y_true):
    y_pred = []
    
    for file in audio_files:
        speech_pred, speech_probs = predict_audio(file)
        text_pred, text_probs = predict_text(file)

        # Fusion step
        fused_pred = fusion_prediction(speech_pred, speech_probs, text_pred, text_probs)
        y_pred.append(fused_pred)

    return y_pred

# Example Usage
long_audio_file = "./combined_model/validation_test/Ses03F_script01_3.wav"  # Input audio file
audio_files, y_true = split_audio(long_audio_file, df)

# Get fused predictions
y_pred = batch_fusion_predictions(audio_files, y_true)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Save confusion matrix as an image
plt.show()

# Print classification report
report = classification_report(y_true, y_pred, target_names=class_labels.values())
print(report)