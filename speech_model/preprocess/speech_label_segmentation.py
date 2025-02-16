import pandas as pd
import os
from pathlib import Path

pd.options.mode.copy_on_write = True

# path to input dir
input_dir = "./speech_model/labels/sorted"

# path to output dir
output_dir = "./speech_model/labels/by_segment"

# Define the segment duration
segment_duration = 6.0

# Create a list to store the new labels

def parse_csv(file_path):
    df = pd.read_csv(file_path)
    segmented_labels = []

    min_time = 0.0
    max_time = df["end_time"].max()
    k = 0

    current_time = min_time

    while current_time + segment_duration <= max_time:
        segment_start = current_time
        segment_end = current_time + segment_duration
        filename = Path(file_path).stem + f"_{k}"

        # Find rows that overlap with this segment
        overlapping = df[(df["start_time"] < segment_end) & (df["end_time"] > segment_start)]

        if not overlapping.empty:
            # Compute dominant emotion (the one covering most of the segment)
            overlapping["overlap_duration"] = overlapping.apply(
                lambda row: min(segment_end, row["end_time"]) - max(segment_start, row["start_time"]),
                axis=1
            )
            dominant_emotion_row = overlapping.loc[overlapping["overlap_duration"].idxmax()]
            dominant_emotion = dominant_emotion_row["emotion"]

            # Average VAD values
            avg_valence = overlapping["valence"].mean()
            avg_arousal = overlapping["arousal"].mean()
            avg_dominance = overlapping["dominance"].mean()

            # Store the segment information
            segmented_labels.append({
                "filename": filename,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "emotion": dominant_emotion,
                "valence": avg_valence,
                "arousal": avg_arousal,
                "dominance": avg_dominance
            })

        k += 1
        # Move to the next segment 
        current_time += segment_duration

    return pd.DataFrame(segmented_labels)


for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)
        output_csv = Path(file_path).stem + ".csv"
        session_data = parse_csv(file_path)

        session_data.to_csv(os.path.join(output_dir, output_csv), index=False)
        print(f"Saved segmented labels to {output_csv}")
        
