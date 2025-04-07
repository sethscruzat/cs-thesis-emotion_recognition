import os
import pandas as pd
from collections import Counter

def count_emotions_in_folder(folder_path, emotion_column='emotion'):
    emotion_counts = Counter({'positive': 0, 'negative': 0, 'neutral': 0})
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                if emotion_column in df.columns:
                    counts = df[emotion_column].value_counts()
                    for emotion in ['positive', 'negative', 'neutral']:
                        emotion_counts[emotion] += counts.get(emotion, 0)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return dict(emotion_counts)

csv_folder_path = "./speech_model/labels_2"
counts = count_emotions_in_folder(csv_folder_path, emotion_column='emotion')
print(counts)