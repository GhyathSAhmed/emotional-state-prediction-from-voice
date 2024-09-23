import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np

def Creat_dataFrame(data_dir):

    # List to store parsed information (actor_id, sentence_id, emotion_id, strength_id, file_path)
    data_list = []

    # Function to parse the filename and extract metadata
    def parse_filename(file_name):
        parts = file_name.split('_')
        actor_id = parts[0]  # Actor ID
        sentence_id = parts[1]  # Sentence ID
        emotion_id = parts[2]  # Emotion ID
        strength_id = parts[3].split('.')[0]  # Remove .mp3 from strength ID
        return actor_id, sentence_id, emotion_id, strength_id

    # Iterate through all the MP3 files in the directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.mp3'):  # Only process MP3 files
            # Parse the filename
            actor_id, sentence_id, emotion_id, strength_id = parse_filename(file_name)
            
            # Full file path for audio loading later
            file_path = os.path.join(data_dir, file_name)
            
            # Append data to list
            data_list.append([actor_id, sentence_id, emotion_id, strength_id, file_path])

    # Convert the list to a Pandas DataFrame for better organization
    df = pd.DataFrame(data_list, columns=['actor_id', 'sentence_id', 'emotion_id', 'strength_id', 'file_path'])

    # # Save the DataFrame to a CSV file
    # df.to_csv('parsed_audio_data.csv', index=False)

    # Show the first few rows of the DataFrame
    print('df is done')

    return df

    ########
def Audio_MFCC_convertion(data_frame):
    # Function to extract MFCC features from an MP3 file
    def mp3_to_mfcc(file_path):
        y, sr = librosa.load(file_path, sr=16000)  # Load audio and resample to 16kHz
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCCs
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Take the mean across time for each MFCC coefficient
        return mfccs_mean

    # Initialize an empty list to store MFCC features
    mfccs_list = []

    # Loop through all file paths and extract MFCC features
    for file_path in data_frame['file_path']:
        mfccs = mp3_to_mfcc(file_path)
        mfccs_list.append(mfccs)

    # Convert the list of MFCCs to a NumPy array for model input
    X = np.array(mfccs_list)

    print(f"Features shape (MFCCs): {X.shape}")

    return X




