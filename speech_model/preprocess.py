import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def split_audio_to_segments(file_path, segment_length=6): #change segment length as needed

    audio = AudioSegment.from_file(file_path)
    duration = len(audio) // 1000  # Duration in seconds
    segments = []
    
    for start_time in range(0, duration, segment_length):
        end_time = min(start_time + segment_length, duration)
        segment = audio[start_time * 1000:end_time * 1000]
        segments.append(segment)
    
    return segments

def process_audio_segment(segment, sr=22050, n_mels=128, output_dir="output_spectrograms", file_prefix=""):
    # STFT parameters
    n_fft = int(0.025 * sr)  # 25ms window
    hop_length = int(0.01 * sr)  # 10ms offset
    
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
    
    # Save spectrogram as image
    segment_name = f"{file_prefix}_segment_{np.random.randint(1e5)}.png"  # Unique naming
    plt.savefig(os.path.join(output_dir, segment_name))
    plt.close()

def process_audio_folder(folder_path, segment_length=6, output_dir="output_spectrograms"):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".wav", ".mp3")):  # Check file type
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Split file into segments
                segments = split_audio_to_segments(file_path, segment_length)
                
                # Process each segment
                for i, segment in enumerate(segments):
                    process_audio_segment(segment, output_dir=output_dir, file_prefix=f"{os.path.splitext(file)[0]}_{i}")

#change later
input_folder = "path_to_your_audio_folder"

#process_audio_folder(input_folder, segment_length=6, output_dir="output_spectrograms")