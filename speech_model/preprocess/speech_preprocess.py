import os
from pydub import AudioSegment
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import cv2

input_folder = "./dataset/5/wav"
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
    """Apply noise reduction using noisereduce."""
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

    mel_spectrogram_db = librosa.power_to_db(mel_spec, ref=np.max)
    # mel_spec_normalized = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX)
    # mel_spec_image = mel_spec_normalized.astype(np.uint8)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    specific_output_folder = output_dir + file_prefix
    os.makedirs(specific_output_folder, exist_ok=True)

    segment_name = f"{file_prefix}_{j}.png"  # Unique naming
    # cv2.imwrite(os.path.join(specific_output_folder, segment_name), mel_spectrogram_db)
    
    librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (db)')

    # # Plot the Delta features
    # plt.subplot(3, 1, 2)
    # librosa.display.specshow(delta_mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    # plt.title('Delta (First Derivative)')
    # plt.colorbar()

    # # Plot the Delta-Delta features
    # plt.subplot(3, 1, 3)
    # librosa.display.specshow(delta2_mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    # plt.title('Delta-Delta (Second Derivative)')
    # plt.colorbar()

    plt.tight_layout()
    
    # Save spectrogram as image
    plt.savefig(os.path.join(specific_output_folder, segment_name))
    plt.close()

def process_audio_folder(folder_path, segment_length=9, output_dir=output_folder):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".wav", ".mp3")):  # Check file type
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Split file into segments
                segments = split_audio_to_segments(file_path, segment_length)
                
                # Process each segment
                for i, segment in enumerate(segments):
                    process_audio_segment(segment, output_dir=output_dir, file_prefix=f"{os.path.splitext(file)[0]}", j = i)

# BATCH PROCESSING
process_audio_folder(folder_path=input_folder,output_dir=output_folder)

# # SINGLE FILE PROCESSING
# # Split file into segments
# segments = split_audio_to_segments("./dataset/wav/Ses05F_impro01.wav")

# # Process each segment
# for i, segment in enumerate(segments):
#     process_audio_segment(segment, output_dir="./speech_model/output_spectrograms/", file_prefix=f"{os.path.splitext("Ses05F_impro01.wav")[0]}", j = i)