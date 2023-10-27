import numpy as np
import h5py
from scipy.io import wavfile



# Open the HDF5 file

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load your HDF5 dataset
with h5py.File('D:/big data/practice/big_data_project/EPIC_audio.hdf5', 'r') as hdf:
    
    data = hdf.get('P01_01')
    dataset1 = np.array(data)
    print("attributes ", data.attrs)

# Normalize the dataset (assuming it's audio data)
normalized_dataset = (dataset1 / np.max(np.abs(dataset1))).astype(np.float32)

# Create a time array for x-axis
sample_rate = 44100  # Sample rate of the audio
time = np.arange(0, len(normalized_dataset)) / sample_rate

# Create a new figure
plt.figure(figsize=(10, 4))

# Plot the audio waveform
plt.plot(time, normalized_dataset, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')

# Display the plot
plt.show()

"""
# Normalize the dataset (assuming it's audio data)
normalized_dataset = (dataset1 / np.max(np.abs(dataset1))).astype(np.float32)

# Create a time array for x-axis
sample_rate = 44100  # Sample rate of the audio
time = np.arange(0, len(normalized_dataset)) / sample_rate

# Create a new figure
plt.figure(figsize=(10, 4))

# Plot the audio waveform
plt.plot(time, normalized_dataset, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')

# Display the plot
plt.show()
"""


"""

import h5py
import numpy as np

# Open the HDF5 file for reading
hdf5_file = h5py.File('D:/big data/practice/project/EPIC_audio.hdf5', 'r')

# Assuming you have multiple datasets inside the HDF5 file
dataset_names = list(hdf5_file.keys())  # Get a list of dataset names
print(hdf5_file.keys())

# Create a function to load audio data from a specific dataset
def load_audio_from_hdf5(dataset_name):
    audio_data = np.array(hdf5_file[dataset_name])
    return audio_data

 #Load the first 10 audio datasets and add them to a list
num_datasets_to_load = 10
audio_data_list = [load_audio_from_hdf5(dataset_name) for dataset_name in dataset_names[:num_datasets_to_load]]


# Close the HDF5 file when you're done
hdf5_file.close()
"""


"""
import h5py

# Open the original HDF5 file for reading
original_hdf5_file = h5py.File('D:/big data/practice/project/EPIC_audio.hdf5', 'r')

# Create a new HDF5 file for writing
new_hdf5_file = h5py.File('smaller_audio_data.hdf5', 'w')

# Assuming you have multiple datasets inside the original HDF5 file
dataset_names = list(original_hdf5_file.keys())  # Get a list of dataset names

# Specify the number of datasets to copy to the new file
num_datasets_to_copy = 10

# Iterate over the first 10 dataset names and copy the datasets to the new file
for dataset_name in dataset_names[:num_datasets_to_copy]:
    original_dataset = original_hdf5_file[dataset_name]
    
    # Create a corresponding dataset in the new file and copy data
    new_dataset = new_hdf5_file.create_dataset(dataset_name, data=original_dataset[:])

# Close both HDF5 files
original_hdf5_file.close()
new_hdf5_file.close()
"""

"""

turn to lstm:


def create_model():
    model = Sequential()

    # Input shape is (40,)
    model.add(Dense(128, input_shape=(40,)))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(64))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))

    model.add(Dense(num_labels)) 
    model.add(Activation('softmax'))
    
    model.compile(optimizer='adam',  loss='categorical_crossentropy', metrics=['accuracy'])

    return model
"""