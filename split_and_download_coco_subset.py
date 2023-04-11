import pycocotools.coco as coco
import matplotlib.pyplot as plt

import json
import os
import random


print(80*"*")

# this script - 
# --- creates a list of desired image indices
# --- filters the annotation file to a shorter annotation file
# --- downloads the images to /images (need to uncomment the line)


# PREP:
# --- download .json annotation file from cluster
# --- place it with this script in the destination
# --- change user parameters
# --- define image_idx_list

###########################################################################################
# USER PARAMETERS

path = "/home/itamar/Desktop/test/continuous_trip/"
json_name = "192_3d_asu_train_FULL.json"

# download_command_file_name = "download_commands.sh"
# short_annotation_file_name = "itamar_short_annotation_file.json"
short_annotation_file_name = "192_3d_asu_train_500.json"
cluster_name = "cluster_CenterNet"

# pick a sample list to take
image_idx_list = list(range(500))
# image_idx_list = random.sample(range(450000,600000),3000)
# image_idx_list = [5,400230, 587290]

###########################################################################################

'''
def get_coco_subset(dataset, img_idx):
    """
    returns smaller coco dataset with image only from img_idx
    """
    pass

def prep_rsync_download_list(dataset, output_folder):
    """"
    return a list of rsync commands for downloading all images in 'dataset'
    """"
    pass
    
subset = get_coco_subset(data, range(10,200))
'''


###########################################################################################


json_path = path + json_name
image_target_path = '/home/itamar/Desktop/test/continuous_trip/images/'

# open existing annotation json
'''JSON'''
# with open(json_path, 'r') as f:
# 	data = json.load(f)

'''COCO ALTERNATIVE:'''
coco_ds = coco.COCO(json_path)


# how many samples are in it?
#print(len(data['images'])+' samples in the original json)')
# view structure:
#print(data['images'][0])

'''JSON'''
# short_list = [data['images'][ix] for ix in image_idx_list]

'''COCO ALTERNATIVE:'''
short_list = coco_ds.loadImgs(image_idx_list)
print(f'short_list: {short_list}')

# create output data structure
new_data = {"images":[], "annotations":[], "categories":[]}

with open(path+short_annotation_file_name, 'w+') as short_annotation_file:
	short_annotation_file.truncate(0)
	for ix, img_info in enumerate(short_list):
		
		print(f'image: {ix} out of {len(short_list)}')

		# execute download command
		cmd = f'rsync -az {cluster_name}:{img_info["file_name"]} {image_target_path}'
		# TODO - uncomment to download the image from cluster
		# os.system(cmd)

		
		#print(f'ix: {ix},\nimg_info: {img_info}')
		# print(f'cmd: {cmd}')

		# take its data as-is, and update its path
		new_data['images'].append(img_info)
		new_path = os.path.join(path, os.path.basename(img_info['file_name'].strip()))
		new_data['images'][ix]['file_name'] = new_path
	
	print(f'new_data: {new_data}')
	print(f'new_data["images"]: {new_data["images"]}')

	# take only annotations matching images in our list
	'''JSON'''
	# for annotation_idx in range(len(data["annotations"])):
	# 	if(data["annotations"][annotation_idx]['image_id'] in image_idx_list):
	# 		new_data["annotations"].append(data["annotations"][annotation_idx])
	'''COCO ALTERNATIVE:'''
	annotation_ids = coco_ds.getAnnIds(image_idx_list)
	annotations = coco_ds.loadAnns(annotation_ids)
	coco_ds.showAnns(annotations)

	
	new_data["annotations"] = annotations

	# take all category information as-is
	'''JSON'''
	# if "categories" in data.keys():
	# 	new_data["categories"] = data["categories"]
	'''COCO ALTERNATIVE'''
	all_cats = list(range(1,len(coco_ds.cats)+1))
	new_data["categories"] = coco_ds.loadCats(all_cats)
	# TODO - use coco_ds.getCatIds+LoadCats to take only the relevant categories and not all

	# write the new data into a new annotation file
	new_data_object = json.dumps(new_data, indent=4)
	short_annotation_file.write(new_data_object)
