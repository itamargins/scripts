import json
import pandas as pd
import glob
import os
import argparse
import numpy as np
from ast import literal_eval
import math

# %%
cognata_class_conversion = {
    'Pedestrian':                               {'imagry_name': 'person',                    'imagry_id': 1},
    'Rider':                                    {'imagry_name': 'person',                    'imagry_id': 1},
    'Pedestrians':                              {'imagry_name': 'person',                    'imagry_id': 1},
    # TODO - use this?
    'Workers':                                  {'imagry_name': 'person',                    'imagry_id': 1},

    'Bicycle':                                  {'imagry_name': 'bicycle',                   'imagry_id': 2},
    'EU_Bicycle_only':                          {'imagry_name': 'bicycle',                   'imagry_id': 2},
    'EU_Pedestrian_bicycle_shared_use':         {'imagry_name': 'bicycle',                   'imagry_id': 2},

    'Car':                                      {'imagry_name': 'car',                       'imagry_id': 3},
    'Cars':                                     {'imagry_name': 'car',                       'imagry_id': 3},

    'Motorcycle':                               {'imagry_name': 'motorbike',                 'imagry_id': 4},
    'Delivery_Bike':                            {'imagry_name': 'motorbike',                 'imagry_id': 4},

    'Cones':                                    {'imagry_name': 'cone',                      'imagry_id': 5},
    'Cone':                                     {'imagry_name': 'cone',                      'imagry_id': 5},

    'Bus':                                      {'imagry_name': 'bus',                       'imagry_id': 6},
    'Buses':                                    {'imagry_name': 'bus',                       'imagry_id': 6},

    'Trains':                                   {'imagry_name': 'train',                     'imagry_id': 7}, #need to use sub-category

    'Truck':                                    {'imagry_name': 'truck',                     'imagry_id': 8},
    'Trucks':                                   {'imagry_name': 'truck',                     'imagry_id': 8},

    'Trash_Can_Metal':                          {'imagry_name': 'trash can',                 'imagry_id': 9},
    'Trash_Can_Plastic':                        {'imagry_name': 'trash can',                 'imagry_id': 9},

    # TODO - Props will include other objects that may hurt this class detection. need to use sub-category
    'Traffic_light':                            {'imagry_name': 'traffic light',             'imagry_id': 10},
    'Traffic_Lights':                           {'imagry_name': 'traffic light',             'imagry_id': 10},

    'Sign_IL_NoEntry_SignPlate':                {'imagry_name': 'no entry sign',             'imagry_id': 11},
    'Sign_JP_No_Entry':                         {'imagry_name': 'no entry sign',             'imagry_id': 11},

    'Sign_IL_Stop_SignPlate':                   {'imagry_name': 'stop sign',                 'imagry_id': 12},

    'Sign_IL_NoUTurn (1)':                      {'imagry_name': 'no u turn sign',            'imagry_id': 13},
    'EU_No_uTurn':                              {'imagry_name': 'no u turn sign',            'imagry_id': 13},

    'Sign_IL_PedestrianCrossing_SignPlate':     {'imagry_name': 'pedestrain caution sign',   'imagry_id': 14},
    'EU_Pedestrian_Crossing':                   {'imagry_name': 'pedestrain caution sign',   'imagry_id': 14},
    'EU_warning_pedestrian_crossing':           {'imagry_name': 'pedestrain caution sign',   'imagry_id': 14},

    'Dog_Big':                                  {'imagry_name': 'dog',                       'imagry_id': 15},

    'Fence':                                    {'imagry_name': 'fence',                     'imagry_id': 16},
    'Pole':                                     {'imagry_name': 'fence',                     'imagry_id': 16},
    'PoleElectricLight':                        {'imagry_name': 'fence',                     'imagry_id': 16},
    'Barricades':                               {'imagry_name': 'fence',                     'imagry_id': 16},
    'Barrier':                                  {'imagry_name': 'fence',                     'imagry_id': 16},
    # TODO - is this right?
    'Traffic Signal Poles':                     {'imagry_name': 'fence',                     'imagry_id': 16},
    'Traffic_Lights_Poles':                     {'imagry_name': 'fence',                     'imagry_id': 16},

    'Van':                                      {'imagry_name': 'van',                       'imagry_id': 17},
    'Vans':                                     {'imagry_name': 'van',                       'imagry_id': 17},
    'Generic_Van':                              {'imagry_name': 'van',                       'imagry_id': 17},

    'TrafficSign':                              {'imagry_name': 'signs',                     'imagry_id': 18}
}


# %%
# helper functions
# ----------------------------------------------------------------------


def list_uknown_appearing_classes(id_list):
    uknown_appearing_classes = set()
    objects_file = os.path.join(cognata_data_dir,'objects.csv')
    with open(objects_file,'r') as objects_file:
        objects = pd.read_csv(objects_file)
    for id in id_list:
        uknown_appearing_classes.add((objects[objects['id']==id]['semanticType.Label']).values[0])
    return uknown_appearing_classes

def sort_error_dict_for_print(error_dict):
    sorted_error_dict = {}
    for (k,v) in error_dict.items():
        sorted_error_dict[k] = sorted(list(v))
    return sorted_error_dict

def convert_cognata_class(possible_classes, convert_to):
    cognata_class_details = None
    # decide on single class_name
    if isinstance(possible_classes, list):
        # iterate over possible categories from specific to broad
        for possible_class in reversed(possible_classes):
            if cognata_class_details is not None:
                break
            cognata_class_details = cognata_class_conversion.get(possible_class)

    # elif isinstance(possible_classes, str):
    #     # print('recieved str')
    #     cognata_class_details = cognata_class_conversion.get(possible_classes)
    
    
    # take desired data attribute
    if cognata_class_details is not None:
        return cognata_class_details.get(str(convert_to))
    return None


def convert_cognata_bbox_to_coco_bbox(cognata_bbox):
    cognata_bbox = literal_eval(cognata_bbox)
    cognata_bbox = [int(value) for value in cognata_bbox]
    width = cognata_bbox[2] - cognata_bbox[0]
    height = cognata_bbox[3] - cognata_bbox[1]
    return [cognata_bbox[0], cognata_bbox[1], width, height]

def get_calibration_info(sensor):
    # desired info
    info = ['resolution', 'fov', 'cam_intrinsics', 'mounting_position']

    calibration_file = os.path.join(cognata_data_dir,'calibration.json')
    with open(calibration_file,'r') as calibration_file:
        calibration_data = json.load(calibration_file)
        calibration_data = calibration_data['cameras']['sensor_id' == sensor]
        
        # use the info list to read these attributes from file
        info = list(map(calibration_data.get, info))
    return info

def create_annotation_id_set():
    annotation_id_set = set()
    # annotation_file_list = os.listdir(os.path.join(cognata_data_dir, sensor+'_ann'))
    camera_ann_list = list(sorted(glob.glob(os.path.join(camera_ann_dir, '*.csv'))))
    for annotation_file in camera_ann_list:
        objects_ids = set((pd.read_csv(annotation_file)['object_ID']).values)
        annotation_id_set.update(objects_ids)
    return annotation_id_set

def read_objects_file():
    # read object_ID from all annotations, and then read only these id's from objects.csv
    annotation_id_set = create_annotation_id_set()
    objects_file = os.path.join(cognata_data_dir,'objects.csv')
    with open(objects_file,'r') as objects_file:
        objects = pd.read_csv(objects_file)
    filtered_objects = objects[objects['id'].isin(annotation_id_set)]

    object_id_to_possible_classes_dictionary = {}
    for i in range(len(filtered_objects)):       
        # can we use type(brandID) == string?
        # ===> apparently not (there are also real objects which don't have brandID. Sumeet needs to answer.
        # for now, filtering according to annotated ID's
        if(isinstance(filtered_objects.iloc[i]['brandID'], str)):
            possible_classes_list = filtered_objects.iloc[i]['brandID'].split('/')
            possible_classes_list.append(filtered_objects.iloc[i]['semanticType.Label'])
        elif math.isnan(filtered_objects.iloc[i]['brandID']):
            possible_classes_list = [filtered_objects.iloc[i]['semanticType.Label']]
        object_id_to_possible_classes_dictionary.update( {filtered_objects.iloc[i]['instance.Number']:possible_classes_list} )

    return object_id_to_possible_classes_dictionary

def calculate_bbox_center(list_of_points):
    # assume x's are in even indices and y's in odd indices
    num_points = len(list_of_points)/2
    x_total = sum(list_of_points[::2])
    y_total = sum(list_of_points[1::2])
    return [x_total/num_points, y_total/num_points]


def pixel_x_coordinate_to_azimuth(pixel_x_coordinate, image_width, fov, focal_length=None):
    IMAGE_WIDTH=image_width
    fov = np.deg2rad(fov)
    pixelNumber = pixel_x_coordinate
    if focal_length is None:
        focal_length = 1/((2/IMAGE_WIDTH)*np.tan(fov/2))
    azimuth_rad = np.arctan((pixelNumber-(IMAGE_WIDTH/2))/(focal_length))
    # azimuth_deg = np.rad2deg(azimuth_rad)
    return azimuth_rad


# %%
# parse arguments
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(description='convert cognata GT to coco format for CenterNet object detection training')
parser.add_argument('--data_dir', '-d',
                    dest = 'data_dir', 
                    type = str,
                    help = 'root directory. includes: $SENSOR_ann/, $SENSOR_jpg/, calibration.json, objects.csv'
                    )
parser.add_argument('--sensor', '-s',
                    dest = 'sensor',
                    type = str,
                    default = 'FrontCam01',
                    help = 'desired sensor annotation'
                    )
# parser.add_argument('--output', '-o',
#                     dest = 'output location',
#                     type = str,
#                     required = False,
#                     help = ''
#                     )
args = parser.parse_args()

cognata_data_dir = args.data_dir
sensor = args.sensor
camera_ann_dir = os.path.join(cognata_data_dir,sensor+'_ann')
images_dir = os.path.join(cognata_data_dir,sensor+'_jpg')
out_path = os.path.join(cognata_data_dir, 'coco_converted_gt.json')

# list csv's
# ----------------------------------------------------------------------
camera_ann_list = list(sorted(glob.glob(os.path.join(camera_ann_dir, "*.csv"))))
ret = {'images': [], 'annotations': [], 'categories': []}


######################### DEBUG
step = []
# y = [[],[],[]]
facing_angle = []
location2 = []
rot = []
c_dist = []
center_cam = []
centerx_3d=[]
centery_3d=[]
az_3d=[]
az_2d=[]
manual_az=[]
######################### DEBUG


# for each annotation file, open, read, process, write the data to dict, and append to total data
# ----------------------------------------------------------------------
annotation_counter = 0

resolution, fov, cam_intrinsics, mounting_position = get_calibration_info(sensor)
object_id_to_possible_classes_dictionary = read_objects_file()

error_dict = {
    'annotated but no brand_ID' : set(),
    'unknown class to convert' : set()
}

# %%
#########################
### FRAME LOOP start
#########################
for camera_ann_file in camera_ann_list:
    df = pd.read_csv(camera_ann_file)

    if df.empty:
        print(f'No detections in {os.path.basename(camera_ann_file)}!')
        frame_id = int(os.path.basename(camera_ann_file).split('.')[0])
    else:
        frame_id = int(df['frame_ID'][0])

    image_info = {'file_name': '{}/{}.jpg'.format(images_dir,str(frame_id).zfill(10)),
                'id': int(frame_id),
                'camera_type': sensor,
                'fov': int(fov['horizontal']),
                'calibration': cam_intrinsics
                }
    ret['images'].append(image_info)

    if df.empty: # TODO - if we don't want to include images with no annotations, raise this to start of loop
        continue

    # %%
    #########################
    ### ANNOTATION LOOP start
    #########################

    for annotation_idx in range(len(df)):
        # calculate new attributes
        # ----------------------------------------------------------------------
        
        # increment first and assign
        annotation_id = (annotation_counter:=annotation_counter+1)

        object_id = df['object_ID'][annotation_idx]
        possible_classes = object_id_to_possible_classes_dictionary.get(object_id)
        if possible_classes is None:
            print(f'{object_id = }, NO BRAND_ID ---> NOT CONVERTED')
            error_dict['annotated but no brand_ID'].add(object_id)
            continue
        cat_id = convert_cognata_class(possible_classes,'imagry_id')
        if cat_id is None:
            print(f'{object_id = }, UNKNOWN CLASS TO CONVERT ---> NOT CONVERTED ({possible_classes})')
            error_dict['unknown class to convert'].add(object_id)
            continue
        print(f'{object_id = }, {possible_classes = }, \n\
              converted to: {convert_cognata_class(possible_classes,"imagry_name")}')

        # df: width (x), height (y), length (z)
        dim = literal_eval(df['dimensions'][annotation_idx])
        # coco: height (y), width (x), length (z)
        dim = [dim[1], dim[0], dim[2]]
        dim = [round(d,2) for d in dim]

        coco_bbox = convert_cognata_bbox_to_coco_bbox(df['bounding_box_2D'][annotation_idx])

        depth = literal_eval(df['center_cam'][annotation_idx])[2]

        location = literal_eval(df['center_cam'][annotation_idx])
        # cognata: y=0 is an object the same height as the camera. positive = down, negative = up
        # CenterNet: y=0 is camera height. positive = up, negative = down

        rotation_y = np.deg2rad(literal_eval(df['rotation_cam'][annotation_idx])[1]-90)
        # assert angle in [-pi, pi] interval
        rotation_y = np.arctan2(np.sin((rotation_y)), np.cos((rotation_y)))

        list_of_3d_bbox_vertices = ['bounding_box_3D_frontTopLeft', 'bounding_box_3D_frontTopRight',
                                    'bounding_box_3D_frontBottomLeft', 'bounding_box_3D_frontBottomRight',
                                    'bounding_box_3D_rearTopLeft', 'bounding_box_3D_rearTopRight',
                                    'bounding_box_3D_rearBottomLeft', 'bounding_box_3D_rearBottomRight']
        bbox_3d_points = []
        # bbox_3d_points += literal_eval(df['bounding_box_3D_frontTopRight'][annotation_idx])
        for x in list_of_3d_bbox_vertices:
            # print(x)
            bbox_3d_coordinates = (df[x][annotation_idx])
            if not(bbox_3d_coordinates != bbox_3d_coordinates):
                bbox_3d_points += literal_eval(df[x][annotation_idx])
            else:
                # TODO - solve this: either by projecting, or putting some placeholder (depends on if azimuth_3d will be relevant)
                pass
            # print(f'{bbox_3d_points = }')
        # bbox_3d_points += [literal_eval(df[x][annotation_idx]) for x in list_of_3d_bbox_vertices]
        # bbox_3d_points = [item for sublist in bbox_3d_points for item in sublist] # used to merge sublists to main list
        bbox_3d_center = calculate_bbox_center(bbox_3d_points)
        azimuth_3d = pixel_x_coordinate_to_azimuth(bbox_3d_center[0], resolution['width'], fov['horizontal'], focal_length=None)

        alpha = rotation_y - azimuth_3d

        bbox_2d_center = calculate_bbox_center(literal_eval(df['bounding_box_2D'][annotation_idx]))
        azimuth_2d = pixel_x_coordinate_to_azimuth(bbox_2d_center[0], resolution['width'], fov['horizontal'], focal_length=None)

        azimuth_shift = azimuth_3d - azimuth_2d

        cx_3d = bbox_3d_center[0]
        cy_3d = bbox_3d_center[1]

        min_distance = df['min_distance'][annotation_idx]

        truncated = df['truncation'][annotation_idx] if not math.isnan(df['truncation'][annotation_idx]) else -1
        
        occluded = df['occlusion'][annotation_idx] if not math.isnan(df['occlusion'][annotation_idx]) else -1

        # %%
        # drop annotations that are too occluded
        # TODO - is this how we want to handle this? what should be the threshold?
        if occluded > 0.90:
            continue

        ann = {
            'image_id': frame_id,
            'id': annotation_id,
            'category_id': cat_id,
            'dim': dim,
            'bbox': coco_bbox,
            'depth': depth,
            'alpha': alpha,
            'location': location,
            'rotation_y': rotation_y,
            'azimuth_3d': azimuth_3d,
            'azimuth_2d': azimuth_2d,
            'azimuth_shift': azimuth_shift,
            'image_fov': np.deg2rad(fov['horizontal']),
            'cx_3d': cx_3d,
            'cy_3d': cy_3d,
            'pitch_angle': 0.0,
            'roll_angle': 0.0,
            'min_depth':  min_distance,
            'truncated': truncated, # unnecessary
            'occluded': occluded, # unnecessary
            }
        ret['annotations'].append(ann)


################# DEBUG
        if True:
        # if(frame_id%500 == 0) and (cat_id == 0):
            step.append(frame_id)
            # fcam = literal_eval(df["facing_cam"][annotation_idx])
            # for i in range(3):
            #     y[i].append(np.rad2deg(fcam[i]))
            # facing_angle.append(df['facing_angle'][annotation_idx])
            location2.append(location[2])
            # rot.append((literal_eval(df['rotation_cam'][annotation_idx])[1]))
            # c_dist.append(df['center_distance'][annotation_idx])
            # center_cam.append(literal_eval(df['center_cam'][annotation_idx])[2])
            # centerx_3d.append(cx_3d-(960/2))
            # centery_3d.append(cy_3d-(540/2))
            # az_3d.append(azimuth_3d)
            # az_2d.append(azimuth_2d)
            # manual_az.append(np.rad2deg(np.arctan(location[0]/location[2])))
################# DEBUG



    #########################
    ### ANNOTATION LOOP end
    #########################

#########################
### FRAME LOOP end
#########################

# %%
# copy the category list
# ----------------------------------------------------------------------
prev_cat = ''
for cat in cognata_class_conversion.keys():
    curr_cat = cognata_class_conversion[cat]['imagry_name']
    if curr_cat == prev_cat:
        continue
    new_dict = {'name': cognata_class_conversion[cat]['imagry_name'],
                'id': cognata_class_conversion[cat]['imagry_id']
                }
    ret['categories'].append(new_dict)
    prev_cat = curr_cat


######################### DEBUG
import matplotlib.pyplot as plt
# for i in range(3):
#     plt.plot(step,y[i], label=f'facing_cam[{i}]')
# plt.plot(step, facing_angle, label='facing_angle')
# plt.plot(step, location2, label='location[2]')
# plt.plot(step,rot, label='rotation_cam[1]')
# plt.plot(step, center_cam, label='center_cam')
# plt.plot(step, c_dist, label='center_dist')
# plt.plot(step, centerx_3d, label='centerx_3d')
# plt.plot(step, centery_3d, label='centery_3d')
# plt.plot(step, az_3d, label='az_3d')
# # plt.plot(step, az_2d, label='az_2d')
# # plt.plot(step, manual_az, label='manual_az')
# plt.legend()
# plt.show()
######################### DEBUG



# %%
# write to new file
# ----------------------------------------------------------------------
print("# images: ", len(ret['images']))
print("# annotations: ", len(ret['annotations']))

sorted_error_dict = sort_error_dict_for_print(error_dict)
print(f'ERROR OBJECT ID LIST: ===> {sorted_error_dict} <===')

uknown_appearing_classes = list_uknown_appearing_classes(sorted_error_dict['unknown class to convert'])
print(f'{uknown_appearing_classes = }')

json_object = json.dumps(ret, indent=4)
with open(out_path, "w") as outfile:
    outfile.write(json_object)
