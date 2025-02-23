import os
import pandas as pd
import shutil

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
# directory = "./speech_model/labels/all"
# output_file = "all_labels_new.csv"  # Change this to your desired output file
# merge_csv_files(directory, output_file)

def append_png_to_filenames(csv_file):
    df = pd.read_csv(csv_file)
    if 'filename' in df.columns:
        df['filename'] = df['filename'].astype(str).apply(lambda x: x if x.endswith('.png') else x + '.png')
        df.to_csv(csv_file, index=False)
        print(f"Updated filenames in {csv_file}")
    else:
        print("No 'filename' column found in the CSV file.")

#append_png_to_filenames("./speech_model/labels/all_labels.csv")

# df = pd.read_csv("./speech_model/labels/all_labels.csv")
# unique_values = df["emotion"].unique()
# print(f"Unique values in 'emotion':")
# print(unique_values)

def copy_all_files(source_directory, target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    for root, _, files in os.walk(source_directory):
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_directory, file)
            shutil.copy2(source_file, target_file)
    
    print(f"All files from {source_directory} have been copied to {target_directory}")

# source_directory = "./speech_model/output_spectrograms/"
# target_directory = "./speech_model/all_spectrograms"
# copy_all_files(source_directory, target_directory)

def delete_files_not_in_csv(directory, csv_file):
    df = pd.read_csv(csv_file)
    if 'filename' in df.columns:
        valid_filenames = set(df['filename'].astype(str))
        deleted_files = 0
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file not in valid_filenames:
                    os.remove(os.path.join(root, file))
                    deleted_files += 1
        
        print(f"Deleted {deleted_files} files not listed in {csv_file}")
    else:
        print("No 'filename' column found in the CSV file.")

target_directory = "./speech_model/all_spectrograms"
csv_file = "./speech_model/labels/all_labels.csv"
delete_files_not_in_csv(target_directory, csv_file)