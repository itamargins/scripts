from Constants import *
import numpy as np

MM_HEIGHT = 224
MM_WIDTH = 280
mtx = 5.6

# Params for test
TASK = 'ddd_asu'
confidence = 0.3
vihecles_threshold = 0.25
ctdet_white_list = {i+1:confidence for i in range(number_of_used_classes)}

pixel_to_meter = 0.178571
meter_to_pixel = 1 / pixel_to_meter
pixel_length_in_mm = pixel_to_meter * 1000

car_center = MM_HEIGHT // 2
min_depth = 10000
out_size = 192
vehicles = [2,5,6,7,16] 
big_objects = [2,5,6,7,8,16]
counter = 0
videos_dir = '/home/Data/models/3d_asu_CenterNet/exp/ddd_asu/videos/'

# Factors for resizing
mm_size = (3*MM_WIDTH,3*MM_HEIGHT) #(840,672)
z = 0.
zs = np.array([z, z, z, z]).reshape((4,1))


left_yaw_ = np.deg2rad(76)
right_yaw_ = np.deg2rad(76)
# Distance between the left & center / right & center cameras' lenses
left_lens_distance = 460.0
right_lens_distance = 499.0
# Distance components on the X & Y axis
CL_x = np.sin(left_yaw_) * left_lens_distance
CL_y = np.cos(left_yaw_) * left_lens_distance
CR_x = np.sin(right_yaw_) * right_lens_distance
CR_y = np.cos(right_yaw_) * right_lens_distance
cyaw = 1.

yaw_map = np.array([
        [0,     (-1 * CL_y,     -1 * CL_x,      left_yaw_ * -1)     ],
        [1,     (0.,                   0.,      np.deg2rad(-cyaw))  ],
        [2,     (-1 * CR_y,     +1 * CR_x,      right_yaw_)         ],
        [3,     (-1808.,                   120.,   np.deg2rad(180 + cyaw)) ]
    ], dtype = object)

# This for all images
height, width = 720,1280
trip_paths = '/home/imagry/DepthData/Depth_annotation_data'
d = 3
exp_path = '/home/Data/models/3d_asu_CenterNet/exp/ddd_asu/'

class_names = cats
CLS2ID = { c : i for i,c in enumerate(class_names) }
IDS2CLS = {v:k for k, v in CLS2ID.items()}

# For bigger minimap
const_factor = 2
pad_factor = 10
left = int(pad_factor * meter_to_pixel)
top = 0
mm_width = int(MM_WIDTH*const_factor)
mm_height = int(MM_HEIGHT*const_factor)
big_car_center = mm_height // 2
# MKZ_width = 4.92506
# MKZ_height = 2.05
camera_shift = 0.23
niro_width = 4.355 #meters
niro_height = 1.8 #meters with mirrors
p0 = (int(np.round(left-(niro_width+camera_shift)*meter_to_pixel/2)), int(np.round(big_car_center-niro_height*meter_to_pixel/2)))
p1 = (int(np.round(left+(niro_width-camera_shift)*meter_to_pixel/2)), int(np.round(big_car_center+niro_height*meter_to_pixel/2)))

# Half of 1280,720
cx = 640
cy = 360
line_length = 10000
base_point = (0,0)
v_fov = np.deg2rad(75)
image_fov = np.deg2rad(90)

left_h_fov_point = (int(left-(-line_length * np.sin(image_fov/2)*meter_to_pixel)), int(big_car_center-(line_length * np.cos(image_fov/2)*meter_to_pixel)))
right_h_fov_point = (int(left+(line_length * np.sin(image_fov/2)*meter_to_pixel)), int(big_car_center+(line_length * np.cos(image_fov/2)*meter_to_pixel)))

left_h_fov_line = (-line_length * np.sin(image_fov/2), line_length * np.cos(image_fov/2))
right_h_fov_line = (line_length * np.sin(image_fov/2), line_length * np.cos(image_fov/2))
left_v_fov_line = (-line_length * np.sin(v_fov/2), line_length * np.cos(v_fov/2))
right_v_fov_line = (line_length * np.sin(v_fov/2), line_length * np.cos(v_fov/2))
center_h_fov_line = (0,100) 

BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)
BGR_ORANGE = (0,165,255)
BGR_RED_ORANGE = (0,215,255)

LEFT='left'
RIGHT='right'
CENTER = 'center'
d_pos = 0.10
line_length = 30

world_size_x_m = MM_HEIGHT / mtx
world_size_y_m = MM_WIDTH / mtx

world_cfg={
    'world_size_x': world_size_x_m,
    'world_size_y': world_size_y_m,
    'out_size_x': MM_HEIGHT,
    'out_size_y': MM_WIDTH,
    'h_world_size_x': world_size_x_m/2,
    'world_ratio': mtx
}

big_world_cfg={
    'world_size_x': int(world_size_x_m*const_factor),
    'world_size_y': int(world_size_y_m*const_factor),
    'out_size_x': mm_height,
    'out_size_y': mm_width,
    'h_world_size_x': int(world_size_x_m*const_factor/2),
    'world_ratio': mm_width / int(world_size_y_m*const_factor)
}

img_size = (1280,720)
side_cam_idx = [0,2]



# Given full path
def get_img_trip(img):
    return img.split('/')[-5]

def get_cam_idx(img):
    return img.split('/')[-3]

# camera index for multi formats
cam_idx_map = {0:0, 2:1, 4:2, 5:3}
