import h5py
import shutil

# Specify the paths to the original HDF5 file and the new HDF5 file
original_file_path = 'D:/big data/practice/project/EPIC_audio.hdf5'
new_file_path = 'D:/big data/practice/project/test_data.hdf5'

# Define a list of dataset names to extract from the original HDF5 file
datasets_to_extract = ["dataset1", "dataset2", "dataset3"]

# Open the original HDF5 file in read mode
with h5py.File(original_file_path, 'r') as original_file:
    # Create a copy of the original file
    shutil.copy(original_file_path, new_file_path)

# Open the new HDF5 file in write mode
with h5py.File(new_file_path, 'r+') as new_file:
    # Create a group in the new file to store the extracted datasets
    group = new_file.create_group("/extracted_datasets")

    # Loop through the list of datasets to extract
    for dataset_name in datasets_to_extract:
        if dataset_name in new_file:
            # Copy the dataset to the new file
            new_file.copy(dataset_name, group)

# Open the new HDF5 file one more time to remove the extracted datasets
with h5py.File(new_file_path, 'r+') as new_file:
    # Remove the extracted datasets from the new file
    for dataset_name in datasets_to_extract:
        if dataset_name in new_file:
            del new_file[dataset_name]

# At this point, you have the new HDF5 file with the extracted datasets removed.

# Optionally, you can delete the original HDF5 file if you want to replace it
# with the one that no longer contains the extracted datasets.
shutil.move(new_file_path, original_file_path)
