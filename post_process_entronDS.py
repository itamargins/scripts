import json
import argparse
import os 
import numpy as np

###############################################################

annotation_file_path = '/home/itamar/Desktop/temp/entron_train.json'
image_target_path = '/home/itamar/Desktop/temp/'

camera_config_set = set()
new_images = 0
category_list = list(range(1,200))
full_class_instance_hist = dict.fromkeys(category_list, 0)

user_permission_changeImagePaths = input("CHANGE IMAGES FILE NAME? (y/n): ")
user_permission_downloadImages = input("DOWNLOAD NEW IMAGES FROM CLUSTER? (y/n): ")
user_permission_imagesFov = input("CONVERT IMAGE FOV FROM STRING TO INT? (y/n): ")
user_permission_annotationsFov2rad = input("CONVERT ANNOTATIONS image_fov FROM STRING TO RAD? (y/n): ")

###############################################################

with open(annotation_file_path,'r') as annotation_file:
    data = json.load(annotation_file)


def image_to_annotation_indices(image_id):
    annotations_indices = []
    for index,ann_data in enumerate(data['annotations']):
        if ann_data['image_id'] == image_id:
            annotations_indices.append(index)
    return annotations_indices
def write_image_fov_attribute(image_annotations_indices, fov):
    for image_annotation_index in image_annotations_indices:
        data['annotations'][image_annotation_index]['image_fov'] = np.deg2rad(fov)

###############################################################

for i, image_data in enumerate(data['images']):

    image_target = image_data["file_name"].replace(
        '/home/imagry/DepthData/Depth_annotation_data/',
        image_target_path
    )

    # download image from cluster
    #---------------------------------------
    if user_permission_downloadImages == 'y':
        cmd = f'rsync -az --mkpath cluster_CenterNet:{image_data["file_name"]} {image_target}'
        print(f'{i}/{len(data["images"])}')
        print(image_target)
        if os.path.exists(image_target):
            print('EXISTS')
        
        else:
            os.system(cmd)
            print('DOWNLOADING')
            new_images += 1

    # change image file path in json - ONLY AFTER DOWNLOADING!
    #---------------------------------------
    if user_permission_changeImagePaths == 'y':
        image_data['file_name'] = image_target


    # convert images' fov from string to num
    #---------------------------------------
    if user_permission_imagesFov == 'y':
        image_data['fov'] = int(float(image_data['fov']))
    
    # list cameras in dataset - ONLY IF THEY ARE INTs ALREADY!
    camera_config_set.add(f'{image_data["camera_type"]}_{image_data["fov"]}')

    # convert annotations' fov from string to num - ARCHIVE (relevant only for image_fov value mismatch)
    #---------------------------------------
    # if user_permission_annotationsFov2rad == 'y':
    #     image_annotations_indices = image_to_annotation_indices(image_data['id'])
    #     write_image_fov_attribute(image_annotations_indices, int(float(image_data['fov'])))

###############################################################


for i,ann_data in enumerate(data['annotations']):
    full_class_instance_hist[ann_data["category_id"]] += 1


    # convert annotations' fov from string to num
    # ---------------------------------------
    if user_permission_annotationsFov2rad == 'y':
        ann_data['image_fov'] = int(float(ann_data['image_fov']))
        ann_data['image_fov'] = np.deg2rad(ann_data['image_fov'])
    
    



###############################################################
# write edited json
with open(annotation_file_path,'w') as converted_json:
    json.dump(data,converted_json, indent = "")



###############################################################
# dataset stats
class_instance_hist = {x:y for x,y in full_class_instance_hist.items() if y!=0}
print(f'DONWLOADED {new_images} new images')
print(15*'*', 'DATASET STATS:', 15*'*')
print(f'{len(data["images"])} IMAGES')
print(f'{len(data["annotations"])} OBJECTS')
print(f'{camera_config_set = }')
print(f'{class_instance_hist = }')
print(f'{full_class_instance_hist = }')