import os
from pydub import AudioSegment
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr

def split_audio_to_segments(file_path, segment_length=6): #change segment length as needed
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

def process_audio_segment(segment, sr=22050, n_mels=128, output_dir="./speech_model/output_spectrograms/", file_prefix="", j=0):
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    samples = librosa.util.normalize(samples)  # Normalize
    # STFT parameters
    n_fft = int(0.025 * sr)  # 25ms window
    hop_length = int(0.01 * sr)  # 10ms offset

    y = librosa.resample(samples, orig_sr=segment.frame_rate, target_sr=sr)
    y = reduce_noise(y, sr)
    
    # Compute STFT and Mel spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    mel_spec = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr, n_mels=n_mels)
    
    # Generate spectrogram plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    specific_output_folder = output_dir + file_prefix
    os.makedirs(specific_output_folder, exist_ok=True)
    
    # Save spectrogram as image
    segment_name = f"{file_prefix}_{j}.png"  # Unique naming
    plt.savefig(os.path.join(specific_output_folder, segment_name))
    plt.close()

def process_audio_folder(folder_path, segment_length=6, output_dir="./speech_model/output_spectrograms/"):
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
input_folder = "./dataset/wav/"
process_audio_folder(input_folder)



# # SINGLE FILE PROCESSING
# # Split file into segments
# segments = split_audio_to_segments("./dataset/wav/Ses05F_impro01.wav")

# # Process each segment
# for i, segment in enumerate(segments):
#     process_audio_segment(segment, output_dir="./speech_model/output_spectrograms/", file_prefix=f"{os.path.splitext("Ses05F_impro01.wav")[0]}", j = i)