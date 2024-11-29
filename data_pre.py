import os
import pandas as pd
import librosa
import numpy as np
import timeit
from tqdm import tqdm


    # Function to parse the filename and extract metadata
def parse_filename(file_name):
    parts = file_name.split('_')
    emotion_id = parts[2]  # Emotion ID

    return  emotion_id



def Creat_dataFrame(data_dir):

    # List to store parsed information (actor_id, sentence_id, emotion_id, strength_id, file_path)
    data_list = []

    # Iterate through all the MP3 files in the directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.mp3'):  # Only process MP3 files
            # Parse the filename
            emotion_id = parse_filename(file_name)
            
            # Full file path for audio loading later
            file_path = os.path.join(data_dir, file_name)
            
            # Append data to list
            data_list.append([emotion_id, file_path])

    # Convert the list to a Pandas DataFrame for better organization
    df = pd.DataFrame(data_list, columns=['Emotions', 'path'])

    # # Save the DataFrame to a CSV file
    # df.to_csv('parsed_audio_data.csv', index=False)

    print('df is done')

    return df

###############
## data agumantation functions
###############

# NOISE
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

###########
# features functions
###########

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.mean(zcr)

def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.mean(rmse)

def chroma(data,sr):
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    return chroma

def mel(data,sr):
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    return mel

def constrast(data,sr):
    stft = np.abs(librosa.stft(data))
    constrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    return constrast

def tonnetz(data,sr):
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y=data),
                                              sr=sr).T, axis=0)
    return tonnetz

def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combine = np.hstack((np.mean(mfcc.T, axis=0),np.mean(delta.T, axis=0),np.mean(delta2.T, axis=0)))
    return combine

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                    chroma(data,sr),
                    mel(data,sr),
                    tonnetz(data,sr),
                    constrast(data,sr),
                    mfcc(data,sr,frame_length,hop_length)
                     ))
    return result


# def get_features(path):
#     data,sr=librosa.load(path)
#     aud=extract_features(data)
#     # audio=np.array(aud)
    
#     # noised_audio=noise(data)
#     # aud2=extract_features(noised_audio)
#     # audio=np.vstack((audio,aud2))
    
#     # pitched_audio=pitch(data,sr)
#     # aud3=extract_features(pitched_audio)
#     # audio=np.vstack((audio,aud3))
    
#     # pitched_audio1=pitch(data,sr)
#     # pitched_noised_audio=noise(pitched_audio1)
#     # aud4=extract_features(pitched_noised_audio)
#     # audio=np.vstack((audio,aud4))

#     return aud

data_dir = 'AudioMP3'

data_path = Creat_dataFrame(data_dir)


start = timeit.default_timer()
X,Y=[],[]
for path,emotion,index in tqdm (zip(data_path.path,data_path.Emotions,range(data_path.path.shape[0]))):
    data,sr=librosa.load(path)
    features=extract_features(data)

    if index%500==0:
        print(f'{index} audio has been processed')
    # for i in features:
    X.append(features)
    Y.append(emotion)
print('Done')
stop = timeit.default_timer()

print('Time: ', stop - start)

# print the shape of the features and target
print(len(X), len(Y), data_path.path.shape)

# saving the data
Emotions = pd.DataFrame(X)
Emotions['Emotions'] = Y
Emotions.to_csv('emotions_5.csv', index=False)
Emotions.head()