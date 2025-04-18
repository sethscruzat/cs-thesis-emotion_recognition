import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import cv2
from pydub import AudioSegment

input_folder = "./dataset/5/wav" # change input and output folder as needed
output_folder = "./speech_model/output_spectrograms/5/"

def split_audio_to_segments(file_path, segment_length): #change segment length as needed
    audio = AudioSegment.from_file(file_path)
    duration = len(audio) // 1000  # Duration in seconds
    segments = []

    for start_time in range(0, duration, segment_length):
        end_time = min(start_time + segment_length, duration)
        segment = audio[start_time * 1000:end_time * 1000]
        segments.append(segment)
    
    return segments

def reduce_noise(y, sr):
    # Estimate noise profile from the first 0.5 seconds
    noise_sample = y[:int(sr * 0.5)]
    reduced_y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    return reduced_y

def process_audio_segment(segment, sr=22050, n_mels=128, output_dir=output_folder, file_prefix="", j=0):
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    samples = librosa.util.normalize(samples)  # Normalize

    y = librosa.resample(samples, orig_sr=segment.frame_rate, target_sr=sr)
    y = reduce_noise(y, sr)

    # STFT parameters
    segment_duration = len(y) / sr  # Get actual segment duration in seconds
    n_fft = min(int(segment_duration * sr * 0.025), 2048)  # 25ms window, cap at 2048
    hop_length = max(1, int(n_fft / 2))  # Ensure meaningful stride
    
    # Compute STFT and Mel spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')

    mel_spec = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Generate spectrogram plot
    plt.figure(figsize=(10, 4))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    specific_output_folder = output_dir + file_prefix
    os.makedirs(specific_output_folder, exist_ok=True)

    segment_name = f"{file_prefix}_{j}.png" 
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (db)')

    plt.tight_layout()
    
    # Save spectrogram as image
    plt.savefig(os.path.join(specific_output_folder, segment_name))
    plt.close()

def process_audio_folder(folder_path, segment_length=9, output_dir=output_folder):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".wav", ".mp3")):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Splits file into multiple segments
                segments = split_audio_to_segments(file_path, segment_length)
                
                # Processes each segment
                for i, segment in enumerate(segments):
                    process_audio_segment(segment, output_dir=output_dir, file_prefix=f"{os.path.splitext(file)[0]}", j = i)

# BATCH PROCESSING
process_audio_folder(folder_path=input_folder,output_dir=output_folder)