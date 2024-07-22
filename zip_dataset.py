import os
import pandas as pd
import argparse
import json
import glob
from lib.opts import parse_list
import zipfile
from tqdm import tqdm

"""
zip dataset files with relevant images (preserving hierarchy) 
    in order to upload to S3 and use independently from cluster
"""

#%%
#TODO - import this in some way from upload_data_to_cloud (requires arguments)
def read_images_list(dataset_file):
    if dataset_file.endswith('json'):
        with open(dataset_file, 'r') as f:
            images = json.load(f)['images']
            filenames = [x['file_name'] for x in images]
        return filenames
    elif dataset_file.endswith('csv'):
        filenames = pd.read_csv(dataset_file)['file_name'].tolist()
        return filenames
    elif os.path.isdir(dataset_file):
        train_files = glob.glob(dataset_file+ "/*train_images*")
        val_files = glob.glob(dataset_file+ "/*val_images*")
        if len(train_files) and len(val_files):
            print(f"found images train/val files: \n{train_files[0]}\n{val_files[0]}")
            return read_images_list(train_files[0]) + read_images_list(val_files[0])


def create_archive(file_list, archive_name):
  """
  Creates a zip archive with preserved hierarchy.

  Args:
    file_list: List of file paths to be included in the archive.
    archive_name: Name of the archive to be created.
  """
  with zipfile.ZipFile(archive_name, "w") as archive:
    with tqdm(total=len(file_list), desc="Creating archive") as pbar:
        for filename in file_list:
            # Get relative path to preserve hierarchy in the archive
            archive.write(filename, arcname=filename)
            pbar.set_description(f"Processing: {filename}")  # Update status with current file
            pbar.update(1)  # Update progress bar for each file


#%%
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', required=True,
                                 help='comma-separated paths to datasets to zip')
parser.add_argument('--output', required=True,
                                help='path to output archive. Ends with .zip')
args = parser.parse_args()
args.datasets = parse_list(args.datasets)

#%%
_ds_images=[]
for dataset in args.datasets:
    _ds_images.extend(read_images_list(dataset))
# _ds_images = read_images_list(args.train_dataset) + read_images_list(args.val_dataset)
# _ds_images = [x.replace('/trips', '/trips/.') for x in _ds_images]  # for relative rsync - #TODO: needed for archive?
all_files = _ds_images + args.datasets

#%%
create_archive(all_files, args.output)
print(f"Archive '{args.output}' created successfully!")