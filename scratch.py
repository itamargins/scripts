import os
import numpy as np
import glob
import cv2
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import shutil
import imutils
import pickle
import json
import torch


'''##################################################################################################
##################################################################################################
##################################################################################################'''

# CONCATENATE PANDAS DATAFRAMES

# camera_ann_dir = '/home/itamar/Documents/Cognata/Test Track - IG - Car, Pedestrian/CognataCamera_ann/'
# camera_ann_list = list(sorted(glob.glob(os.path.join(camera_ann_dir, "*.csv"))))
# df = pd.concat((pd.read_csv(f).head(1) for f in camera_ann_list), ignore_index=True)
# frames = df['frame_ID']
# rotations = df['rotation_cam']
# rotation_list = [literal_eval(rotations[i])[1] for i in range(len(rotations))]

# x_step = 1300
# plt.plot(frames, rotation_list)
# xs = [x_step*i for i in range((list(frames)[-1]//x_step)+1)]
# plt.xticks(xs)
# plt.ylim(-190,190)
# plt.show()


'''##################################################################################################
##################################################################################################
##################################################################################################'''

# INCREMENT AND ASSIGN A VARIABLE

# x=4
# a = (x:=x+1)
# print(a)
# print(x)



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# CONVERTING PIXEL COORDINATES TO AZIMUTH FROM CAMERA CENTER-RAY
# NOTE: MIGHT NOT BE THE RIGHT ONE. EXPORTED coding_cheatbook CODE MIGHT BE BETTER
# focalLength=700.0
# IMAGE_WIDTH=1280.0
# fov = np.deg2rad(85)
# # fov = 2.0*np.arctan(IMAGE_WIDTH/(2.0*focalLength))
# pixelNumber = 640

# focal_length_itamar = 1/((2/IMAGE_WIDTH)*np.tan(fov/2))

# # azimuth_rad = 2.0*np.arctan(pixelNumber/(2*focalLength)) - (fov / 2.0)
# azimuth_rad = np.arctan((pixelNumber-(IMAGE_WIDTH/2))/(focal_length_itamar)) 

# azimuth_deg = np.rad2deg(azimuth_rad)
# print(azimuth_deg)


'''##################################################################################################
##################################################################################################
##################################################################################################'''

# DOWNLOAD IMAGES FROM CLUSTER USING 

# image_list_file = open('/home/imagry/offline_data/sheba_images/image_list.txt', 'r')

# target_path = '/home/imagry/offline_data/sheba_images/'
# cluster_name = 'imagry@192.168.1.94'
# for image_name in image_list_file:
#     image_name = image_name.rstrip('\n')
#     cmd = f'rsync -az {cluster_name}:/home/Data/entron_rect/idx0/rectified_cust93_noRat/{image_name} {target_path}'
#     os.system(cmd)
# images = os.listdir()



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# root_dir = '/home/itamar/Documents/Cognata/Test Track - IG - Car, Pedestrian/'
# output_dir = '/home/itamar/Documents/Cognata/Test Track - IG - Car, Pedestrian/CognataCamera_jpg/general_test_predictions'
# input = os.path.join(root_dir,'CognataCamera_jpg')
# output = output_dir
# batch_size = 1

# cmd = f'python \
# /opt/imagry/CenterNet/src/general_test.py \
# --task 1 \
# --model /opt/imagry/aidriver_new/models/Detector/2022-11-19-512-288-fastdla_int8_2619bfb/3DASU.pth \
# --input "{input}" \
# --batch_size {batch_size} \
# --output "{output}" \
# --local_test \
# --use_post_process'

# # execute the command
# os.system(cmd)




'''##################################################################################################
##################################################################################################
##################################################################################################'''

# WRITE EMPTY ANNOTATION FILE

# new_dict = {
#     'images':[],
#     'annotations': [],
#         "categories": [
#         {"name": "person","id": 1},
#         {"name": "bicycle","id": 2},
#         {"name": "car","id": 3},
#         {"name": "motorbike","id": 4},
#         {"name": "cone","id": 5},
#         {"name": "bus","id": 6},
#         {"name": "train","id": 7},
#         {"name": "truck","id": 8},
#         {"name": "trash can","id": 9},
#         {"name": "traffic light","id": 10},
#         {"name": "no entry sign","id": 11},
#         {"name": "stop sign","id": 12},
#         {"name": "no u turn sign","id": 13},
#         {"name": "pedestrain caution sign","id": 14},
#         {"name": "dog","id": 15},
#         {"name": "fence","id": 16},
#         {"name": "van","id": 17},
#         {"name": "signs","id": 18}
#     ]
# }
# with open('/home/itamar/Desktop/inter_cognata_jsons/0sims.json','w') as output:
#     json.dump(new_dict,output, indent = "")


### EXAMPLE 3 - ITERATIVELY CATTING COCO DATASETS
# root = '/home/imagry/offline_data/cognata/'
# directory_list = sorted([os.path.join(root, file) for file in os.listdir(root) if os.path.isdir(os.path.join(root, file))])

# for index,directory in enumerate(directory_list):
#     print(80*'*','\n', f'{directory = }, file {index+1}/{len(directory_list)}')

# # CATTING COCO ANNOTATIONS TO LARGE ANNOTATION FILE
#     cmd = f'python /opt/imagry/CenterNet/src/tools/cat_coco_datasets.py \
# -j \
# {f"/home/itamar/Desktop/inter_cognata_jsons/{index}sims.json"} \
# {os.path.join(root, directory,"coco_converted_gt.json")} \
# -o \
# {f"/home/itamar/Desktop/inter_cognata_jsons/"} \
# -t \
# {f"{index+1}sims"} \
# '

# # REPLACING STRING IN ALL ANNOTATION FILES
# #     cmd = f'python /home/itamar/Documents/scripts/replace_string_in_json.py \
# # -i {os.path.join(root, directory,"coco_converted_gt.json")} \
# # -o {os.path.join(root, directory,"coco_converted_gt.json")} \
# # -r itamar20230413 \
# # -w cognata \
# # '

#     # os.system(cmd)
#     with open(f"/home/itamar/Desktop/inter_cognata_jsons/{index+1}sims.json", 'r') as accumulated_json:
#         data = json.load(accumulated_json)
#         print(f'ACCUMULATED {len(data["images"])} IMAGES SO FAR')



# with open('/home/imagry/offline_data/cognata/cognata_103sim.json','r') as full_json:
#     data = json.load(f# with open('/home/imagry/offline_data/sheba_trips/entron_train_converted.json','w') as converted_json:
#     json.dump(data,converted_json, indent = "")ull_json)
# sim_set = set()
# for image_data in(data['images']):
#     sim_set.add(os.path.dirname(image_data['file_name']))
# for directory in directory_list:
#     print(directory+'/FrontCam01_jpg' in sim_set)



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# COUNT NUMBER OF FILES WITH EXTENSION IN DIRECTORY, RECURSIVE
# def fileCount(path, extension):
#     count = 0
#     for root, dirs, files in os.walk(path):
#         print(f'{root = }, {dirs = }, {files = }')
#         count += sum(f.endswith(extension) for f in files)
#     return count

# print(fileCount(root, 'jpg'))



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# CAT IMAGES IN DIR TO A SINGLE IMAGE

# directory = '/home/itamar/Desktop/loc[1]_examples'
# font = cv2.FONT_HERSHEY_SIMPLEX 
# # image_list = sorted(os.listdir(directory))
# # image_list = ['annotation_minus04.jpg', 
# #               'annotation_minus2.jpg', 
# #               'annotation_is_zero.jpg', 
# #               'annotation_no_change.jpg', 
# #               'annotation_plus2.jpg', 
# #               'annotation_plus4.jpg']

# for i,imp in enumerate(image_list):
#     img = cv2.imread(os.path.join(directory,imp))
#     img = imutils.resize(img, width=900)
#     cv2.putText(img,str(imp),(50,50), font, 0.5,(245,245,245),1,cv2.LINE_AA)
#     cv2.imshow('img',img)
#     cv2.waitKey()
#     if i==0:
#         full_image = img.copy()
#     else:
#         full_image = np.concatenate((full_image, img),axis=0)
#     cv2.imshow('full_image',full_image)
#     cv2.waitKey()
 
# cv2.imwrite(os.path.join(directory,'full_image.jpg'), full_image)



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# COMPARE TWO MODELS
'''
modelA = pickle.load(open('/home/itamar/Desktop/modelA.pkl', 'rb'))
modelB = pickle.load(open('/home/itamar/Desktop/modelB.pkl', 'rb'))
print(modelA == modelB)
for p1, p2 in zip(model.parameters(), modelB.parameters()):
    print('sizes MATCH' if p1.size() == p2.size() else "***** sizes DON'T MATCH *****")
    print('values MATCH' if ((p1 == p2).sum() == len(p1.flatten())) else "***** values DON'T MATCH *****")


# manually set weight:
with torch.no_grad():
    modelB.base.base_layer[0].weight[0][0][0][0] = torch.tensor([-0.0857])
'''



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# COMPARE TWO PICKLES

# single_head_dep_zed_90_output = pickle.load(open('/home/itamar/Desktop/1head_dep_zed90_weights.pkl', 'rb'))
# two_head_dep_zed_90_output = pickle.load(open('/home/itamar/Desktop/2head_dep_zed90_weights.pkl', 'rb'))

# for p1, p2 in zip(single_head_dep_zed_90_output.cpu().parameters(), two_head_dep_zed_90_output.cpu().parameters()):
#     print(f'{p1.flatten().size() = }')
#     print(f'{p2.flatten().size() = }')
#     print((p1==p2).sum())
#     print(80*'*')

# a=5



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# OPEN JSON AND READ SPECIFIC FIELD
# with open('/home/imagry/offline_data/sheba_trips/entron_train.json','r') as original_json:
#     data = json.load(original_json)

# # brand_ID_list = {}
# for i in (data['images']):
#     # brand_ID_list.append((i['brandID']))
#     # brand_ID_list.update({i['brandID']:i['properties']["userInput"]['category2']})
#     original = i['fov']
#     conv = int(float(original))
#     print(f'replacing {original = } with {conv = }')
#     i['fov'] = conv
# print(brand_ID_list)

# with open('/home/imagry/offline_data/sheba_trips/entron_train_converted.json','w') as converted_json:
#     json.dump(data,converted_json, indent = "")



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# WORK WITH ANNOTATION FILE - OPERATIONS
# annotation_file_path = '/home/imagry/offline_data/sheba_trips/entron8311/entron8311.json'
# image_target_path = 
# target = '/home/imagry/offline_data/first_train_data/entron_10samples.json'

# desired_type = 'entron'

# from collections import defaultdict
# camera_config_dict = defaultdict(int) # set as keys, count as values
# entron_image_ids = set()

# with open(annotation_file_path,'r') as annotation_file:
#     data = json.load(annotation_file)

# # new_images = 0

# res = {
#     "images":[],
#     "annotations":[],
#     "categories":[]
# }
# for i, image_data in enumerate(data['images']):
    
    
    # visualize annotations for entire DS
#     os.system(f'python /home/itamar/Documents/scripts/visualize_annotation.py --image_path {image_data["file_name"]}')

    # image_data['id'] = i
    # image_data['iscrowd'] = 0
    # res['images'].append(image_data)
#     # convert images' fov from string to num
#     #---------------------------------------
#     image_data['fov'] = int(float(image_data['fov']))
    
    # grab images with desired camera_type
    # if image_data['camera_type'] == desired_type:
    #     res['images'].append(image_data)
    #     entron_image_ids.add(image_data['id'])

    # if image_data['fov'] == 130:
    #     os.system(f'python /home/itamar/Documents/scripts/visualize_annotation.py --image_path {image_data["file_name"]}')
        
  
# #     # list camera configurations
    # camera_config_dict[f'{image_data["camera_type"]}_{image_data["fov"]}'] += 1

# #     # download image from cluster
# #     #---------------------------------------
#     image_target = image_data["file_name"].replace(
#         '/home/imagry/DepthData/Depth_annotation_data/',
#         image_target_path
#     )
#     cmd = f'rsync -az --mkpath cluster_CenterNet:{image_data["file_name"]} {image_target}'
#     print(f'{i}/{len(data["images"])}')
#     print(image_target)
#     if os.path.exists(image_target):
#         print('EXISTS')
      
#     else:
#         os.system(cmd)
#         print('DOWNLOADING')
#         new_images += 1

# indices = list(range(174,192))
# images = {num: [] for num in indices}
# images_with_multiple_indices = []

# image_id_set = set()
# for i,ann_data in enumerate(data['annotations']):

    # find images with single annotation
    # image_id = ann_data['image_id']
    

    # add score 1.0
    # ann_data['score'] = 1.0
    # ann_data['iscrowd'] = 0
    # ann_data['area']=int(ann_data['bbox'][2]*ann_data['bbox'][3])
    # res['annotations'].append(ann_data)

#     # convert annotations' fov from string to num
#     # ---------------------------------------
    # ann_data['image_fov'] = int(float(ann_data['image_fov']))
    # ann_data['image_fov'] = np.deg2rad(ann_data['image_fov'])
#     print(f'{ann_data["location"][2]}')
#     ann_data['location'][2] += 15
#     print(f'{ann_data["location"][2]}')

    # grab annotations with desired camera_type
    # if ann_data['image_id'] in entron_image_ids:
    #     res['annotations'].append(ann_data)
    
    # grab annotations with specific categories
    # if 174 <= ann_data['category_id'] <= 191:
    #     image_id_set.add(ann_data['image_id'])
    #     res['annotations'].append(ann_data)

        # if ann_data['image_id'] in images[ann_data['category_id']]:
        #     images_with_multiple_indices.append(ann_data['image_id'])
        #     print(f'FOUND AN IMAGE WITH MULTIPLE ANNOTATIONS! {ann_data["image_id"]}')
        # else:
        #     images[ann_data['category_id']].append(ann_data['image_id'])


# res['categories'] = data['categories'] 

# print(f'{images_with_multiple_indices = }')
# write edited json
# with open(target,'w') as converted_json:
#     json.dump(res,converted_json, indent = "")

# print(f'DONWLOADED {new_images} new images')
# print(f'{camera_config_dict = }')




'''##################################################################################################
##################################################################################################
##################################################################################################'''

# # DS statistics
# annotation_file_path = '/home/imagry/offline_data/sheba_trips/entron4683/entron4683_FULL.json'
# with open(annotation_file_path,'r') as annotation_file:
#     data = json.load(annotation_file)

# def image_to_annotation_indices(image_id):
#     annotations_indices = []
#     for index,ann_data in enumerate(data['annotations']):
#         if ann_data['image_id'] == image_id:
#             annotations_indices.append(index)
#     return annotations_indices
# def write_image_fov_attribute(image_annotations_indices, fov):
#     for image_annotation_index in image_annotations_indices:
#         data['annotations'][image_annotation_index]['image_fov'] = np.deg2rad(fov)

# camera_set = set()
# category_list = list(range(1,200))
# full_class_instance_hist = dict.fromkeys(category_list, 0)

# for image_data in data['images']:
    # list cameras in dataset
    # camera_set.add(f'{image_data["camera_type"]}_{image_data["fov"]}')
  
#     # IF NEEDED TO RE-WRITE IMAGE_FOV IN THE ANNOTATIONS
#     # image_annotations_indices = image_to_annotation_indices(image_data['id'])
#     # write_image_fov_attribute(image_annotations_indices, int(float(image_data['fov'])))
# x_location = []
# z_location = []
# for ann_data in data['annotations']:
# #     # count category instances
# #     full_class_instance_hist[ann_data["category_id"]] += 1

#     x_location.append(ann_data['location'][0])
#     z_location.append(ann_data['location'][2])
# class_instance_hist = {x:y for x,y in full_class_instance_hist.items() if y!=0}
# print(f'\n{os.path.basename(annotation_file_path)} STATISTICS:')
# print(f'{len(data["images"])} IMAGES')
# print(f'{len(data["annotations"])} OBJECTS')
# print(f'{camera_set = }')
# print(f'{class_instance_hist = }')
# print(f'{full_class_instance_hist = }')
# # with open(annotation_file_path,'w') as converted_json:
# #     json.dump(data,converted_json, indent = "")


# fig, ax = plt.subplots()
# bins = list(range(-100,101,1))
# H,xedges,yedges,im = ax.hist2d(x_location,z_location,bins=bins, cmap='plasma')

# plt.imshow(H, interpolation='nearest', origin='lower',
#         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
#         )
# fig.colorbar(im,location='bottom')
# ax.set_xlim(-75,75)
# ax.set_ylim(0,80)
# # plt.plot(x_location, [abs(v) for v in x_location])
# # plt.plot(x_location, np.abs(x_location))
# from textwrap import wrap

# # ax.set_title(wrap(f'{os.path.basename(annotation_file_path)} - Object Location Histogram', 20))
# title_text = f'{os.path.basename(annotation_file_path)} - \nObject Location Histogram'
# ax.set_title(title_text)
# fig.tight_layout()

# x = np.linspace(-75,75,1000)
# y = np.abs(x)
# ax.plot(x,y,color='white')

# plt.show()

'''##################################################################################################
##################################################################################################
##################################################################################################'''

# USER INPUT - ALSO LOOP UNTIL RIGHT
# print('this is a user input test')
# test_string = input("enter a character: ")
# print(f'your string was: {test_string}')
# while(test_string) != 'y':
#     print(f'NOT doing {test_string}')
#     test_string = input("enter a character: ")
# print(f'doing {test_string}')



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# GENERAL 2D HISTOGRAM

# x = [10,10,20,20,20,50,50,50,50,50,10]
# y = [2*v for v in x]

# # H,xedges,yedges= np.histogram2d(x,y)
# fig, ax = plt.subplots()
# H,xedges,yedges,im = ax.hist2d(x,y)

# plt.imshow(H, interpolation='nearest', origin='lower',
#         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='rainbow',
#         )
# fig.colorbar(im)
# plt.show()



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# DRAW RECTS IN COLORS ON AN IMAGE
# imp = '/home/imagry/offline_data/cognata/new_sims/cognata_sims_20230423_part01_itamar/6444cf894dac90003046b157/FrontCam01_jpg/0000000033.jpg'

# img = cv2.imread(imp)

# signs_bboxes = [
# [666, 360, 672, 366],
# [818, 342, 837, 361],
# [435, 345, 453, 363],
# [798, 359, 805, 366],
# [657, 360, 663, 366],
# [819, 360, 823, 366],
# ]

# colors = [
# (0, 0, 255),    #Red:
# (0, 255, 0),    #Green:
# (255, 0, 0),    #Blue: 
# (0, 255, 255),    #Yellow
# (128, 0, 128),    #Purple
# (0, 165, 255),    #Orange
# ]

# ids = [
#     225,
#     231,
#     232,
#     233,
#     234,
#     241,
# ]

# font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale = 1
# color = (0, 0, 255) # BGR color (red)
# thickness = 2
# lineType = cv2.LINE_AA

# location = [50,50]
# for index,sign_bbox in enumerate(signs_bboxes):
#     cv2.rectangle(img, (sign_bbox[0], sign_bbox[1]), 
#                   (sign_bbox[2], sign_bbox[3]), colors[index], 2)
#     cv2.putText(img, str(ids[index]), tuple(location), font, fontScale,colors[index],thickness,lineType)
#     location[1] += 40

# img = np.flip(img, axis=2)
# plt.imshow(img)
# plt.show()





'''##################################################################################################
##################################################################################################
##################################################################################################'''

# ACCESSING AND ASSIGINING MODEL_DICT (from nikolai)
# (strict = True    : assert that keys between dicts are identical)
# (strict = False   : assign keys that are identical, ignore the rest)


# def replace_dict_keys(self, dic, keys_to_replace={}):
#     for kr in keys_to_replace:
#         keys0 = list(dic.keys())
#         for k0 in keys0:
#             if kr in k0:
#                 dic[k0.replace(kr,keys_to_replace[kr])] = dic.pop(k0)
#     return dic
 
# def load_model_dict(self, model_path=None, strict=True, ignore_list=[], device=None):
#     replace_dict = {
#     'base_drn' : 'base'
#     }
#     ignore_dict = {x: 'IGNORE_THIS' for x in ignore_list}
#     if model_path is not None:
#         try:
#             state_dict = torch.load(model_path, map_location=device)
#             state_dict = self.replace_dict_keys(state_dict, replace_dict)
#             state_dict = self.replace_dict_keys(state_dict, ignore_dict)
#             self.net_module.load_state_dict(state_dict, strict=strict)
#             self.model_path = model_path
#             del state_dict
#         except Exception as e:
#             print('failed to load existing model from {}'.format(model_path), flush=True)
#             print('error : ', e, flush=True)



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# EXAMPLE OF CENTERNET "NMS"

# import torch
# import torch.nn as nn

# def _nms(heat, kernel=3):
#     pad = (kernel - 1) // 2
    
#     hmax = nn.functional.max_pool2d(
#         torch.tensor(heat), (kernel, kernel), stride=1, padding=pad)
#     axarr[1,0].imshow(hmax)
#     keep = (hmax == heat).float()
#     axarr[1,1].imshow(keep)
#     return heat * keep


# heatmap = np.random.rand(50,50)
# heatmap = torch.tensor(np.stack((heatmap,)*3, axis=-1))
# f, axarr = plt.subplots(2,2)
# axarr[0,0].imshow(heatmap)

# heat = _nms(heatmap)
# plt.show()

'''##################################################################################################
##################################################################################################
##################################################################################################'''

# SEABORN PLOTS - INCOMPLETE

# import seaborn as sns


# def create_heatmap(dataset):
#     # Convert the COCO dataset to a Pandas DataFrame
#     df = pd.DataFrame.from_records(dataset['annotations'])

#     # Merge the DataFrame with the category information
#     df = df.merge(pd.DataFrame.from_records(dataset['categories']),
#                   how='left', left_on='category_id', right_on='id')

#     # Group the bounding boxes by category and compute histograms
#     histograms = df.groupby('name')['bbox'].apply(lambda x: pd.Series(x.apply(lambda b: b[2] * b[3]).hist()))

#     # Plotting the heatmaps for each category
#     for category, hist in histograms.items():
#         plt.figure()
#         sns.heatmap(hist.values.reshape(1, -1), cmap='YlOrRd',
#                     xticklabels=False, yticklabels=[category], cbar=False)
#         plt.title(f'Bounding Box Histogram for {category}')
#         plt.xlabel('Bins')
#         plt.ylabel('')
#         plt.show()

# coco_path = "/home/imagry/offline_data/full_zed_ds_2023-05-02/192_3d_asu_train.json"
# with open(coco_path, "r") as f:
#     data = json.load(f)
# print(data.keys())

# with open(coco_path, 'r') as file:
#     coco_data = json.load(file)

# # Create the heatmaps for bounding box histograms
# create_heatmap(coco_data)



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# CREATE 'SUBSET' DUMMY ANNOTATION FILE, GIVEN A FRAME OF INTEREST


# dir = '/home/imagry/offline_data/sheba_trips/images/2023-07-24T11_09_00/3d_images/6/left/'
# foi_name = "1690186563.689105.jpeg"
# range = 50

# cam_idx_to_fov ={
#     "0": 93,
#     "1": 93,
#     "2": 110,
#     "3": 58,
#     "4": 25,
#     "5": 93,
#     "6": 93,
#     "7": 130
# }

# frames = sorted(os.listdir(dir))
# id = frames.index(foi_name)

# #########################################
# # PICK RANGE
# short_frame_list = frames[id-range:id+range]
# # end_frame = '1688464479.039706.jpeg'
# # end_idx = frames.index(end_frame)
# # short_frame_list = frames[id:end_idx]
# # TODO - support case where frame is too close to start or end (not enough range)
# #########################################

# res = {
#     "images":[],
#     "annotations":[],
#     "categories":[]
# }
# for i,file in enumerate(short_frame_list):
#     # filter json if already exists
#     if file.split('.')[-1] == 'json':
#         continue
#     f = filter(None,dir.split('/'))
#     cam_idx = list(f)[-2]
#     res['images'].append(
#         {
#             "file_name": os.path.join(dir,file),
#             "id": i,
#             "camera_type": 'entron',
#             "fov": cam_idx_to_fov.get(cam_idx)
#         }
#     )
#     assert res['images'][-1]['fov'] is not None, 'FOV = None ---> cam_idx_to_fov failed'

# # TODO - generalize naming if wanted...
# with open(os.path.join('/home/imagry/results/temp_results/', f'{foi_name.split(".")[:-2]}_pm{range}.json'),'w') as subset_json:
#     json.dump(res, subset_json, indent = "")


'''##################################################################################################
##################################################################################################
##################################################################################################'''

# CREATE 'SUBSET' DUMMY ANNOTATION FILE (for general test) - for all cameras


# cam_idx_to_fov ={
#     "0": 93,
#     "1": 93,
#     "2": 110,
#     "3": 58,
#     "4": 25,
#     "5": 93,
#     "6": 93,
#     "7": 130
# }

# for cam_index in range(8):
#     print(f'{cam_index = }')
#     directory = f'/home/imagry/offline_data/2023-06-15T15_33_37/3d_images/{cam_index}/left/'
    
#     cam_config = 'entron_1'


#     res = {
#         "images":[],
#         "annotations":[],
#         "categories":[]
#     }
#     files = glob.glob(os.path.join(directory, "**/*.jpeg"), recursive=True)
#     for i,file in enumerate(files):
#         res['images'].append(
#             {
#                 "file_name": file,
#                 "id": i,
#                 "camera_type": cam_config.split('_')[0],
#                 "fov": cam_idx_to_fov[file.split('/')[-3]]
#             }
#         )
#     with open(os.path.join(directory, 'subset.json'),'w') as converted_json:
#         json.dump(res,converted_json, indent = "")



# VERSION 2 - #TODO - COMPARE!
# cam_idx_to_fov ={
#     "0": 93,
#     "1": 93,
#     "2": 110,
#     "3": 58,
#     "4": 25,
#     "5": 93,
#     "6": 93,
#     "7": 130
# }

# root = '/home/imagry/offline_data/sheba_trips/images/2023-06-15T15_23_35/3d_images/'
# images_from_all_cameras = []
# for dir in sorted(os.listdir(root)):
#     data = {"images": [], "annotations": [], "categories": []}
#     print(dir)
#     full_frame_list = os.listdir(os.path.join(root,dir+'/left'))
#     # print((full_frame_list[-3:-1]))
#     images_from_camera = full_frame_list[8000:9000]

# # for camera_idx, images_from_camera in enumerate(images_from_all_cameras):

#     for image in images_from_camera:
#         data['images'].append(
#             {
#                 "file_name": os.path.join(root,str(dir)+'/left',image),
#                 "camera_type": 'entron',
#                 "fov": cam_idx_to_fov[str(dir)]
#             }
#         )
#     with open(f'/home/imagry/offline_data/sheba_trips/images/2023-06-15T15_23_35/3d_images/{dir}/frames_8Kto9K.json','w') as output:
#         json.dump(data,output, indent = "")





'''##################################################################################################
##################################################################################################
##################################################################################################'''

# RUN A PYTHON SCRIPT WITH VARIANTS OF ARGUMENT VALUES

# def make_clean_dir(folder, remove=True):
#     try:
#           os.makedirs(folder)
#     except:
#         if remove:
#             shutil.rmtree(folder)
#             os.makedirs(folder)
        

# root_dir = '/home/imagry/offline_data/sheba_trips/images/2023-06-15T15_23_35/3d_images/'
# camera_dirs = glob.glob(os.path.join(root_dir,'*/'))

# for camera_dir in camera_dirs:
#     input = camera_dir+'frames_8Kto9K.json'
#     output = camera_dir.replace('/home/imagry/offline_data/sheba_trips/images/2023-06-15T15_23_35/3d_images/', '/home/imagry/results/model_entron7122/model_entron7122_fullDS/2023-06-15T15_23_35/')
#     gt = camera_dir+"frames_8Kto9K.json"
#     make_clean_dir(output)
#     cmd = f"python \
# /opt/imagry/CenterNet/src/general_test.py \
# --task 1 \
# --model /opt/imagry/CenterNet/models/entron7122/fullDS/model_best_val.pth \
# --input {input} \
# --output {output} \
# --inference_json {gt} \
# --gt_path {gt}"

#     os.system(cmd)



'''##################################################################################################
##################################################################################################
##################################################################################################'''
# # LOADING AND DISPLAYING SAVED TENSORS' IMAGES

# load_tensor = torch.load('/home/imagry/results/temp_results/single_entron_image_debugging/heat_after_NMS.pt')

# # FOR SINGLE IMAGE
# # for i in range(load_tensor[0].shape[0]):
# #     image = load_tensor[0][i].cpu().numpy()
# #     print(i)
# #     plt.imshow(image, cmap = 'gray')
# #     plt.show()


# # FOR MULTIPLE IMAGES
# number_of_images = load_tensor[0].shape[0]
# f,axes= plt.subplots(3,6, figsize=(15,10))
# axes = axes.flatten() # this is to run over 1D index i
# for i in range(number_of_images):
#     image = load_tensor[0][i].cpu().numpy()
#     axes[i].imshow(image.astype(np.float32), cmap='gray')
#     axes[i].set_title(str(i))
# plt.tight_layout()
# plt.show()


'''##################################################################################################
##################################################################################################
##################################################################################################'''

# MATCH INDICES BETWEEN GT AND PREDS
# pred_file_path = '/home/itamar/Documents/scripts/scratch_dir/entron7122_fullDS_preds_on_entron4683val.json'
# gt_file_path = '/home/itamar/Documents/scripts/scratch_dir/entron4683_val_copy.json'
# with open(pred_file_path,'r') as pred_file:
#     pred_data = json.load(pred_file)
# with open(gt_file_path,'r') as gt_file:
#     gt_data = json.load(gt_file)

# pred_idx_to_gt_idx_map = {}

# # find matching gt and map pred idx to it. keep mapping for annotations later
# for i in range(len(pred_data['images'])):
#     assert pred_data['images'][i]['file_name'] == gt_data['images'][i]['file_name'], \
#         f'file paths ({i}) DO NOT MATCH! cannot use this method...'
    
#     pred_idx_to_gt_idx_map.update(
#         {pred_data['images'][i]["id"] : gt_data['images'][i]["id"]}
#         )
#     pred_data['images'][i]["id"] = gt_data['images'][i]["id"]

# # apply mapping to annotations image_id
# for pred_ann_data in pred_data['annotations']:
#     pred_ann_data['image_id']= pred_idx_to_gt_idx_map.get(pred_ann_data['image_id'])

# with open(pred_file_path,'w') as output:
#     json.dump(pred_data, output, indent = "")




# # ###### 
# root = '/home/imagry/offline_data/cognata/evaluation_sims/Test_Track'
# res_path = '/home/imagry/offline_data/cognata/evaluation_sims/Test_Track/Test_Track.json'
# files = sorted(glob.glob(os.path.join(root, "*.jpg")))

# res = {'images':[]}

# for idx,file in enumerate(files):
#     res['images'].append({
#         "file_name": file,
#         "id": idx,
#         "camera_type": 'zed',
#         "fov": 85
#     })

# with open(res_path,'w') as out:
#     json.dump(res,out, indent="")



##############



'''##################################################################################################
##################################################################################################
##################################################################################################'''

# CLEAN DATASET ACCORDING TO TRIP - REMOVE AND VALIDATION SET

# import json

# with open('/home/imagry/offline_data/sheba_trips/entron_23-08/entron_23-08_LOCAL.json','r') as data_file:
#     data = json.load(data_file)
# new_data = {}
# val_data = {}

# trips_to_remove = [
#     '2023-03-07T13_01_10',
#     '2023-03-07T11_27_20',
#     '2023-03-07T11_20_29',
# ]

# trips_to_val = [
#     '2023-06-02T11_25_45',
#     '2023-03-02T15_25_57',
#     '2023-02-23T13_40_11',
# ]

# removed_indices = []
# removed_image_counter = 0
# val_indices = []
# val_image_counter = 0

# for image in data['images']:
#     image_trip = image['file_name'].split('/')[-5]
#     if image_trip in trips_to_remove:
#         removed_indices.append(image['id'])
#         removed_image_counter += 1
#     if image_trip in trips_to_val:
#         val_indices.append(image['id'])
#         val_image_counter+=1


# new_data['images'] = [image for image in data['images'] if image['id'] not in removed_indices]
# new_data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] not in removed_indices]
# new_data['categories'], new_data['cameras_info'] = data['categories'], data['cameras_info']

# val_data['images'] = [image for image in data['images'] if image['id'] in val_indices]
# val_data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in val_indices]
# val_data['categories'], val_data['cameras_info'] = data['categories'], data['cameras_info']


# with open('/home/imagry/offline_data/sheba_trips/entron_23-08/entron_23-08_filtered_train.json','w') as train_data_file:
#     json.dump(new_data,train_data_file, indent = "")

# with open('/home/imagry/offline_data/sheba_trips/entron_23-08/entron_23-08_filtered_val.json','w') as val_data_file:
#     json.dump(val_data,val_data_file, indent = "")

'''##################################################################################################
##################################################################################################
##################################################################################################'''
# LOAD MODEL'S STATE DICT ONLY


import torch
import sys
import os

sys.path.append('/opt/imagry/CenterNet/src')
sys.path.append('/opt/imagry/CenterNet/src/lib')
model_path = '/opt/imagry/CenterNet/models/sheba_kia_260923_phase2/model_130/model_130.pth'

out_dir = os.path.dirname(model_path)
out_name,ext = os.path.splitext(os.path.basename(model_path))

cp = torch.load(model_path)
print(f'{cp.keys() = }')
# trainable_params = sum(p.numel() for p in cp['state_dict'].parameters() if p.requires_grad)
# print(f'{trainable_params = }')
out_path = os.path.join(out_dir, out_name+'_state_dict'+ext)
torch.save(cp['state_dict'], out_path)
print(f'{out_path = }')