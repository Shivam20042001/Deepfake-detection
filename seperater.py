import os
import shutil
import json

# Define paths
metadata_path=r'Path to metadata file'
video_dir='Path to downloaded videos'
dataset_dir='dataset'
real_dir=os.path.join(dataset_dir,'real')
fake_dir=os.path.join(dataset_dir,'fake')

# Created Dir
os.makedirs(real_dir,exist_ok=True)
os.makedirs(fake_dir,exist_ok=True)

# Load MetaData 

with open(metadata_path,'r')as f:
    metadata=json.load(f)


for filename, info in metadata.items():
    label=info.get('label','').lower()
    source_path=os.path.join(video_dir,filename)

    if label=='real':
        target_path=os.path.join(real_dir,filename)
    else:
        target_path=os.path.join(fake_dir,filename)

    shutil.move(source_path,target_path)

