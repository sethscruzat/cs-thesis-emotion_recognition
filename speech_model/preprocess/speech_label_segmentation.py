import pandas as pd
import os
from pathlib import Path

pd.options.mode.copy_on_write = True

input_dir = "./speech_model/labels/sorted/1/"
output_dir = "./speech_model/labels/by_segment/1/"

sorted_dir = "./speech_model/labels/sorted/4/"
utterance_dir = "./speech_model/labels/by_utterance/4/"

segment_duration = 3.0 # Define the segment duration

def parse_csv(file_path):
    df = pd.read_csv(file_path)
    segmented_labels = []

    min_time = 0.0
    max_time = df["end_time"].max()
    k = 0

    current_time = min_time
    # iterates by indicated segment length until it reaches end of file
    while current_time + segment_duration <= max_time:
        segment_start = current_time
        segment_end = current_time + segment_duration
        filename = Path(file_path).stem + f"_{k}.png"

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

def sort_by_start_time(file_path, output_csv):
    df = pd.read_csv(file_path)
    df = df.sort_values(by="start_time").reset_index(drop=True)

    df.to_csv(os.path.join(sorted_dir, output_csv), index=False)

for file in os.listdir(utterance_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(utterance_dir, file)
        output_csv = Path(file_path).stem + ".csv"
        sort_by_start_time(file_path, output_csv)

        print(f"Saved sorted labels to {utterance_dir}")