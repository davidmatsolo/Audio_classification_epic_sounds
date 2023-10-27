import pandas as pd
import h5py
from pydub import AudioSegment
import os

# Read the data from the provided CSV file
df = pd.read_csv('D:/big data/practice/big_data_project/data/EPIC_Sounds_train.csv')

# Open the HDF5 file that contains the audio data
hdf5_file = h5py.File('D:/big data/practice/big_data_project/data/EPIC_audio.hdf5', 'r')

# Create a folder to store the audio files
output_folder = 'D:/big data/practice/big_data_project/audio_files'
os.makedirs(output_folder, exist_ok=True)

# Select the first 10 rows from your DataFrame
first_10_rows = df.head(10)

# Loop through the selected rows and extract the audio samples
for index, row in first_10_rows.iterrows():
    start_sample = row['start_sample']
    stop_sample = row['stop_sample']
    audio_data = hdf5_file[row['video_id']][start_sample:stop_sample]

    # Create an AudioSegment from the audio data (assuming it's in a suitable format)
    audio = AudioSegment(
        data=audio_data.tobytes(),
        sample_width=audio_data.dtype.itemsize,
        frame_rate=44100,  # Adjust the frame rate as needed
        channels=1  # Adjust the number of channels as needed (1 for mono, 2 for stereo)
    )

    # Save the audio files in the specified output folder
    audio.export(os.path.join(output_folder, f'{row["annotation_id"]}.wav'), format='wav')

