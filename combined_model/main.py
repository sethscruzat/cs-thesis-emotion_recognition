import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
import pickle
from sklearn.preprocessing import StandardScaler
import assemblyai as aai
from tensorflow.keras.models import load_model

# Load Models
cnn_model = load_model("cnn_six_seconds.keras")  # Load trained CNN model
with open("best_model.pkl", "rb") as f:
    nb_model = pickle.load(f)  # Load trained Naïve Bayes model

# Define Weights (Based on Validation Accuracy)
cnn_weight = 0.52 
nb_weight = 0.86 

# Normalize weights so they sum to 1
total_weight = cnn_weight + nb_weight
cnn_weight /= total_weight
nb_weight /= total_weight

class_labels = {2:"Positive", 1: "Neutral", 0: "Negative"}

# SPEECH MODEL
def reduce_noise(y, sr):
    """Apply noise reduction using noisereduce."""
    # Estimate noise profile from the first 0.5 seconds
    noise_sample = y[:int(sr * 0.5)]
    reduced_y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    return reduced_y

def process_audio_segment(audio_file, sr=22050, n_mels=128, target_size=(128, 256)):
    audio = AudioSegment.from_file(audio_file)
    sample = librosa.util.normalize(audio)  # Normalize

    y = librosa.resample(sample, orig_sr=audio.frame_rate, target_sr=sr)
    y = reduce_noise(y, sr)

    # STFT parameters
    segment_duration = len(y) / sr  # Get actual segment duration in seconds
    n_fft = min(int(segment_duration * sr * 0.025), 2048)  # 25ms window, cap at 2048
    hop_length = max(1, int(n_fft / 2))  # Ensure meaningful stride
    
    # Compute STFT and Mel spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')

    mel_spec = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize spectrogram values to 0-255 (image-like format)
    mel_spec_normalized = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX)

    # Resize to match CNN input
    mel_spec_resized = cv2.resize(mel_spec_normalized, target_size)

    spectogram = np.array(mel_spec_resized).reshape(-1, 128, 256, 1)
    spectogram = spectogram / 255.0

    audio_input = np.expand_dims(spectogram, axis=1) 

    # Get CNN probabilities
    cnn_probabilities = cnn_model.predict(audio_input)[0] 

    return cnn_probabilities


# TEXT MODEL
def speech_to_text(audio_file):
    aai.settings.api_key = "c45ab6cc228640d58dfe8b8b43b712df"

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed")
    
    return transcript

def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove punctuation, numbers, etc.
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    return text

def predict_text(text_for_model):
    # Load the same text vectorizer used in training
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    text = preprocess_text(text_for_model)
    # Convert text to feature vector
    text_vector = vectorizer.transform(text)  # Convert to TF-IDF features

    # Get Naïve Bayes probabilities
    nb_probabilities = nb_model.predict_proba(text_vector)[0]  # (num_classes,)

    return nb_probabilities


# Fusion Function (Weighted Confidence Score)
def fusion_prediction(audio_file, text):
    # Get probabilities from both models
    cnn_probs = predict_audio(audio_file)
    nb_probs = predict_text(text)

    # Compute final weighted score
    final_scores = (cnn_weight * cnn_probs) + (nb_weight * nb_probs)

    # Get final prediction (emotion with highest score)
    final_prediction = np.argmax(final_scores)

    return final_prediction, final_scores


# Example Usage
audio_file = "example.wav"  # Input audio file
text_for_model = speech_to_text(audio_file)

predicted_emotion, confidence_scores = fusion_prediction(audio_file, text_for_model)
print("Final Predicted Emotion:", predicted_emotion)
print("Confidence Scores:", confidence_scores)