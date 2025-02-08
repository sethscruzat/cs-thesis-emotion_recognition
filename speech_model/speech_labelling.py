import os
import re
import pandas as pd
from pathlib import Path

# path to EmoEvaluation folder
emo_eval_folder = "./dataset/EmoEvaluation/"
# path to output dir
output_dir = "./speech_model/labels/"

def parse_emo_eval(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            # scans .txt file for an exact match
            match = re.match(r"\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\S+)\s+(\S+)\s+\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]", line)
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                utterance_id = match.group(3)
                emotion_label = match.group(4)

                # just changes the emotion labels into the three used for this study
                # will be changed if and when tackling more than 3 emotions (positive, neutral, anger/frustration, fear/surprise)
                match emotion_label:
                    case "hap":
                        emotion_label = "positive"
                    case "exc":
                        emotion_label = "positive"
                    case "ang":
                        emotion_label = "negative"
                    case "sad":
                        emotion_label = "negative"
                    case "fru":
                        emotion_label = "negative"
                    case "neu":
                        emotion_label = "neutral"

                valence = float(match.group(5))
                arousal = float(match.group(6))
                dominance = float(match.group(7))

                data.append({
                    "utterance_id": utterance_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "emotion": emotion_label,
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": dominance
                })
    
    return pd.DataFrame(data)

for file in os.listdir(emo_eval_folder):
    if file.endswith(".txt"):
        file_path = os.path.join(emo_eval_folder, file)
        output_csv = Path(file_path).stem + ".csv"
        session_data = parse_emo_eval(file_path)

        session_data.to_csv(os.path.join(output_dir, output_csv), index=False)

# Example: Process an EmoEvaluation file
# Single file processing
# emo_eval_file = os.path.join(emo_eval_folder, "Ses05F_impro01.txt")  # Replace with the correct session file
# emotion_data = parse_emo_eval(emo_eval_file)

# output_csv = "Ses05F_impro01_annotations.csv"
# emotion_data.to_csv(output_csv, index=False)
