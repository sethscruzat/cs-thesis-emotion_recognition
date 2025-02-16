import os
import pandas as pd

def merge_csv_files(directory, output_file):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    df_list = [pd.read_csv(os.path.join(directory, file)) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} CSV files into {output_file}")

# Example usage
# directory = "./speech_model/labels/by_segment"
# output_file = "all_labels.csv"  # Change this to your desired output file
# merge_csv_files(directory, output_file)

def append_png_to_filenames(csv_file):
    df = pd.read_csv(csv_file)
    if 'filename' in df.columns:
        df['filename'] = df['filename'].astype(str) + '.png'
        df.to_csv(csv_file, index=False)
        print(f"Updated filenames in {csv_file}")
    else:
        print("No 'filename' column found in the CSV file.")

# append_png_to_filenames("./speech_model/labels/all_labels.csv")

df = pd.read_csv("./speech_model/labels/all_labels.csv")
unique_values = df["emotion"].unique()
print(f"Unique values in 'emotion':")
print(unique_values)