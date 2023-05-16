import os
import sys
import cv2
import numpy as np
import glob
from os.path import join
import argparse
import time
import pandas as pd 
import matplotlib.pyplot as plt
import json
import random
import torch 
import imutils

from ddd_utils import *



###########################################################################################
# USER PARAMETERS

# COGNATA images
# ============
annotation_file = '/home/imagry/offline_data/cognata/old_before_080523/645a086a4dac90003059f850/coco_converted_gt.json'

id = 1366
id = str(id).zfill(10)
image_path = '/home/imagry/offline_data/cognata/old_before_080523/645a086a4dac90003059f850/FrontCam01_jpg/'+id+'.jpg'
sample_origin = 'cognata'
new_size = (1080,1920)
new_size = (288,1024)
output_path = '/home/imagry/offline_data/cognata/old_before_080523/645a086a4dac90003059f850/annotations/'

number_of_wanted_images = 1000
whole_directory = False


# ZED images
# ============
# annotation_file = '/home/imagry/offline_data/2023-01-05_full_model_2619bfb_exp_BIASED/192_3d_asu_biased_unbiased_COMBINED.json'
# image_path = '/home/imagry/offline_data/2023-01-05_full_model_2619bfb_exp/1553103923.019646.jpeg'
# sample_origin = 'zed'
# output_path = '/home/imagry/offline_data/2023-01-05_full_model_2619bfb_exp_BIASED/'
# new_size = (720,1280)


# ZED images - SHEBA
# ============
annotation_file = '/home/imagry/offline_data/sheba_trips/zed3000_entron2057/zed3000_entron2057_val.json'
image_path = '/home/imagry/offline_data/2023-01-05_full_model_2619bfb_exp/1598917495.907449.jpeg'
sample_origin = 'sheba'
output_path = '/home/itamar/Desktop'
new_size = (1080,1920)



# KITTI images
# # ============
# annotation_file = '/home/imagry/KITTI/annotations/kitti_3dop_train.json'
# image_path = '/home/imagry/KITTI/training/image_2/000532.png'
# # image_path = '/home/imagry/KITTI/training/image_2/003234.png' # TODO - this sample seems like it has bad annotations!
# sample_origin = 'kitti'
# output_path = '/home/imagry/KITTI/'


###########################################################################################




#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************
# FUNCTION ALTERNATIVES

def compute_box_3d(dim, location, rotation_y, pitch = np.deg2rad(0), roll = np.deg2rad(0)):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  cc, ss = np.cos(pitch), np.sin(pitch)
  ccc, sss = np.cos(roll), np.sin(roll)

  R_x = np.array([[1,0,0], [0, cc, -ss], [0, ss, cc]], dtype=np.float32)
  R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  R_z = np.array([[ccc, -sss, 0], [sss, ccc, 0], [0, 0, 1]], dtype=np.float32)

  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  if sample_origin == 'cognata':
    y_corners = [-h/2,-h/2,-h/2,-h/2, h/2,h/2,h/2,h/2]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

  R = np.dot(R_z, np.dot(R_x, R_y))
  corners_3d = np.dot(R, corners) 
  # corners_3d = np.dot(R_z, corners) 
  # corners_3d = np.dot(R_y, corners_3d) 
  # corners_3d = np.dot(R_x, corners_3d) 
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
  return corners_3d.transpose(1, 0)




#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************

# Helper function
def exctract_3d_from_info(image, bird_view, key, val,cam_idx, input_w, input_h, \
    src_img_file = None, trip = None, monodepth_array = None):

    # if val.shape[0] > 0:
    if len(val) > 0:
        for obj in val:
            # score = obj[12]
            # if score > ctdet_white_list[key]:
                # TODO - reread the attributes from the annotation
                # cat_id = key-1
                # alpha = obj[0]
                # bbox = [int(x) for x in obj[1:5]]
                # dims = obj[5:8]
                # locations = obj[8:11]
                # rotation_y = obj[11]
                # azimuth_shift = obj[13]
                # min_depth = obj[14] if len(obj) > 14 else None
                # pitch = obj[17] if len(obj) > 18 else 0
                # roll = obj[18] if len(obj) > 18 else 0
                # cx_2d = bbox[0] + (bbox[2]-bbox[0])/2
                # cy_2d = bbox[1] + (bbox[3]-bbox[1])/2


                cat_id = obj['category_id']
                alpha = obj['alpha']
                bbox = obj['bbox']
                dims = obj['dim']
                locations = obj['location']
                rotation_y = obj['rotation_y']
                azimuth_shift = obj['azimuth_shift']
                min_depth = obj['min_depth']
                pitch = obj['pitch_angle']
                roll = obj['roll_angle']

                # cv2.circle(image, (int(cx_2d),int(cy_2d)), 0, (20,255,255), 2)
                # cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color_classes[int(cat_id)], 3)
                # cv2.putText(image,str(int(score*100)),(bbox[2],bbox[1]), font, 0.5,(245,245,245),1,cv2.LINE_AA)

                # if args.draw_center_3d:
                #     centered_location = locations[0],locations[1]-dims[0]/2,locations[2]
                #     center_3d = np.array([centered_location])
                #     cx_3d,cy_3d = fov_project_to_image_without_calib(center_3d, np.deg2rad(85), input_w = input_w, input_h = input_h)[0]
                #     cv2.circle(image, (int(cx_3d),int(cy_3d)), 0, (70,150,255), 2)

                # NOTE: WRONG! DON'T UNCOMMENT. DONT CORRECTLY OUTSIDE OF FUNCTION
                # box_3d = compute_box_3d(dims, locations, rotation_y, pitch, roll)
                # box_2d = fov_project_to_image_without_calib(box_3d, np.deg2rad(85), input_w = input_w, input_h = input_h)
                # image = draw_box_3d(image, box_2d)

                # Filter far signs
                # if cat_id == 17 and locations[2] > 30:
                #     continue

                # For visualization
                shifted_locations = locations[0],locations[1],locations[2]
                shifted_box_3d = compute_box_3d(dims, shifted_locations, rotation_y, pitch, roll)
                rect = shifted_box_3d.copy()[:4, [0, 2]]
                center_x = (bbox[0] + bbox[2])//2
                min_x = np.min(rect[:,0])
                max_x = np.max(rect[:,0])
                min_y = np.min(rect[:,1])

                # if args.use_post_process:
                if True:
                    # For close objects front cam
                    min_x = np.min(rect[:,0])
                    max_x = np.max(rect[:,0])
                    min_y = np.min(rect[:,1])
                    center_x = (bbox[0] + bbox[2])//2

                    if cat_id in [5,6,7,16] and int(cam_idx) == 1 and int(min_y) < 3:
                        # left side
                        if center_x <= input_w//2:
                            #print(f'++++++++++++++++++++++++++++++ left minDepth = {minDepth} max_x ={max_x}')
                            rect[:,0] += ((-1)*min_depth - max_x)
                        else:
                            #print(f'++++++++++++++++++++++++++++++ right minDepth = {minDepth}, min_x = {min_x}')
                            rect[:,0] += min_depth - min_x

                    # Filter too small objects on the images edges
                    if cat_id in [2,5,6,7,8,16] and (bbox[2]-bbox[0])/input_w < 0.025 and (bbox[0] < 10 or bbox[2] > input_w-10):
                        continue
                    
                    # Tackle problems in side cams
                    # min_point_rect = np.min(rect[:,1])
                    # if cam_idx in [0,2]:
                    #     if monodepth_array is not None:
                    #         bbox[0], bbox[2] = int(fx*bbox[0]), int(fx*bbox[2])
                    #         bbox[1], bbox[3] = int(fy*bbox[1]), int(fy*bbox[3])
                    #         for i in range(len(bbox)):
                    #             if bbox[i] < 0:
                    #                 bbox[i] = 0
                    #         min_depth = np.percentile((scale_factor/monodepth_array[0,0, bbox[1]:bbox[3], bbox[0]:bbox[2]]),20)
                        
                    # For side cams
                    # if cam_idx in [0,2] and min_y < 10:
                    #     rect[:,1] += min_depth - min_point_rect
                    # # For the three cams
                    # if locations[0] > -10 and locations[0] < 10 and cam_idx != 3:
                    #     fix_rect_by_2d_box(dims, rect, bbox.copy(), cam_idx in [0,2,3], image_shape = (input_h, input_w))

                    #     if cat_id in [5,6,7,16]:
                    #         padd_dims(rect, dims)
                    
                            
                poly = get_poly_points(rect, mm=True, world_cfg=big_world_cfg)
                poly = poly.reshape((4,2))
                poly = np.concatenate( (poly, zs) , axis=-1).astype(float)
                # rotate objects wrt camera
                if cam_idx in [0,1,2,3]:
                    tmp_y_pos, tmp_x_pos, tmp_yaw = yaw_map[yaw_map[:, 0] == cam_idx].flatten()[1] 
                    hpr = np.array([tmp_yaw, 0, 0], dtype=float)
                    pos = np.array([tmp_y_pos, tmp_x_pos, 0], dtype=float)
                    poly = transform_poly(poly, hpr, pos.copy(), world_cfg=big_world_cfg)
                    rect = poly.copy()[:, 0:2]
                poly = poly[:, 0:2]
                poly[:, 0] += left
                rect[:, 0] += left

                # Check intersection between object and our car
                # if np.min(poly[:, 0]) < left+4*meter_to_pixel and find_intersection(poly):
                #     print('box error overlap with our car')
                #     print('src_img_file,cat_id,rotation_y, location', src_img_file, cat_id, rotation_y, locations)
                #     #print('poly', poly)
                #     errors_file.write('error box: src_img_file,cat_id,rotation_y, location ' \
                #         + src_img_file+ ' ' + str(cat_id) + ' ' + str(rotation_y) + ' ' + ', '.join(map(str, locations)) + '\n')
                #     #errors_file.write('poly ' + ', '.join(map(str, poly)) + '\n')
                poly = poly.reshape((4, 1, 2)).astype(int)
                #Take the first element only
                bird_view = draw_box_bird_view(bird_view, rect, poly,cam_idx)
    return image, bird_view



def transform_poly(poly_points, to_hpr, to_pos, world_cfg=world_cfg):
    MM_HEIGHT = world_cfg['out_size_x']
    MM_WIDTH = world_cfg['out_size_y']
    pixel_to_meter = 0.178571
    meter_to_pixel = 1 / pixel_to_meter
    pixel_length_in_mm = pixel_to_meter * 1000.
    car_center = MM_HEIGHT // 2
    #rotate objects on left/right cams
    MM_HEIGHT_HALF = MM_HEIGHT / 2
    poly_points[:, 1] -= MM_HEIGHT_HALF

    to_pos /= pixel_length_in_mm
    poly_points = Transform(poly_points, to_pos, to_hpr, device='cpu').T
    poly_points[:, 1] += MM_HEIGHT_HALF

    #cast back to cv2 poly function input
    poly_points = poly_points.astype(int)

    return poly_points


def Transform(pcd, pos, hprRad, inv=False, device='cpu'):
    mat = getNumPyMatrixLight(*hprRad)
    if inv:
        mat = np.linalg.inv(mat)
    if 'cuda' in device:
        a = torch.from_numpy(mat).to(device=device)
        b = torch.from_numpy(pcd.T).to(device=device)
        m = torch.matmul(a,b)
        n = m.to(device='cpu').numpy()
    elif device == 'cpu':
        a = mat
        b = pcd.T
        m = a @ b
        n = m
    t_pcd = (n)

    if inv:
        t_pcd = t_pcd - np.array([pos]).T
    else:
        t_pcd = t_pcd + np.array([pos]).T
    return t_pcd

def Rx(a):
    return np.array([[1,0,0,0],
                    [0,np.cos(a),-np.sin(a),0],
                    [0,np.sin(a),np.cos(a),0],
                    [0,0,0,1]])

def Ry(a): 
    return np.array([[np.cos(a),0,np.sin(a),0],
                    [0,1,0,0],
                    [-np.sin(a),0,np.cos(a),0],
                    [0,0,0,1]])
def Rz(a):
    return np.array([[np.cos(a),-np.sin(a),0,0],
                    [np.sin(a),np.cos(a),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
def getNumPyMatrixLight(h, p, r):
    return (Ry(r) @ Rx(p) @ Rz(h))[:3,:3]

#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************







font = cv2.FONT_HERSHEY_SIMPLEX 

# TODO - support argparsing
# parser = argparse.ArgumentParser(description='parse csv')

# parser.add_argument("--task", dest = 'task', help = 
#                     "test3cams: 0, test image folder: 1, test from json: 2",
#                     default = 0, type = int)


gt_data = json.load(open(annotation_file,'r'))


# image_path_list = random.sample(os.path.split(os.path.split(image_path)[0])[0])
full_image_path_list = [image_path]

# TO LIST MANY IMAGES IN A DIRECTORY:
# # -----------------------------------
if whole_directory:
    directory = os.path.dirname(image_path)
    full_image_path_list = sorted(glob.glob(directory+'/*.jpg'))
    full_image_path_list = full_image_path_list[:number_of_wanted_images]

image_path_list = full_image_path_list


for image_path in image_path_list:
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)

    # sample_origin = 'kitti' if 'KITTI' in image_path else 'zed'

    # generate an empty bird's-eye view image
    bird_view = gen_bird_view_image()

    # translate image full name to id in ds
    # image_id = [gt_data['images'][i]['id'] for i in range(len(gt_data['images'])) if gt_data['images'][i]['file_name']==image_path][0]
    # for i in range(len(gt_data['images'])):
    for image_data in gt_data['images']:
        if image_data['file_name']==image_path:
            image_id = image_data['id']
            # if calibration data is included, take it
            calibration = image_data.get('calibration')
            if calibration is not None:
                # TODO - add a COLUMN of zeros at the end!!! which order are the rows here?
                calibration.append([0,0,0])
                # for i in calibration:
                #     i.append(0)
            break

    # translate image id in ds to matching annotations
    # image_annotations = [gt_data['annotations'][i] for i in range(len(gt_data['annotations'])) if gt_data['annotations'][i]['image_id']== image_id]
    image_annotations = [gt_data['annotations'][i] for i in range(len(gt_data['annotations'])) if gt_data['annotations'][i]['image_id']== image_id]

    print(f'{image_annotations = }')

    calib_kitti = np.array(
        [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
         [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
         [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
        dtype=np.float32)

    calib_zed = np.array(
        [[699,                  0.000000000000e+00, 1280//2,               0],
         [0.000000000000e+00,   699,                720//2,                0],
         [0.000000000000e+00,   0.000000000000e+00, 1.000000000000e+00,    0]],
        dtype=np.float32)

    for ann in image_annotations:
        coco_bbox = ann['bbox']
        locations = ann['location']
        dims = ann['dim']
        rotation_y = ann['rotation_y']
        pitch = ann['pitch_angle']
        roll = ann['roll_angle']
        cat_id = ann['category_id']
        image_fov = ann['image_fov']
        
        # truncated = ann['truncated']
        # occluded = ann['occluded']
        # cv2.rectangle(image,(int(coco_bbox[0]),int(coco_bbox[1])),(int(coco_bbox[0])+ int(coco_bbox[2]),int(coco_bbox[1]) + int(coco_bbox[3])),(20,255,255), 2)




        # debug - print to image
        # TODO - remove
        BR = str(coco_bbox[0]+coco_bbox[2]+5)+','+str(coco_bbox[1]+coco_bbox[3])
        from ast import literal_eval
        BR_values = literal_eval(BR)
        BR_values = tuple(int(s) for s in BR_values)
        TL = str(coco_bbox[0]-5)+','+str(coco_bbox[1])
        TL_values = literal_eval(TL)
        TL_values = tuple(int(s) for s in TL_values)

        str_list = ["{:.2f}".format(num) for num in locations]
        #", ".join(str_list)
        # cv2.putText(image, str((int(coco_bbox[0]),int(coco_bbox[1]))), TL_values, font, 0.4,(0,0,255),1,cv2.LINE_AA)
        # cv2.putText(image, f'center_cam[1]: {locations[1]:.5f}', (500,200), font, 0.4,(0,0,255),1,cv2.LINE_AA)
        second_row = (BR_values[0], BR_values[1]+20)
        # cv2.putText(image, f'prop: {(locations[1]/dims[2]):.5f}', second_row, font, 0.4,(0,0,255),1,cv2.LINE_AA)
        # cv2.putText(image, f'{occluded:.2f}', TL_values, font, 0.4,(0,0,255),1,cv2.LINE_AA)

        #TODO - fix sample_origin business

        if(sample_origin == 'zed'):
            # locations[2] *= -1 
            # locations[0] *= -1
            calib = calib_zed
            # drop last column - #TODO???
            new_calib = []
            for i,row in enumerate(calib):
                new_calib.append(row[:-1])
            calib = np.array(new_calib)
        elif(sample_origin == 'kitti'):
            calib = calib_kitti

        if calibration is not None:
            calib = np.array(calibration)


        box_3d = compute_box_3d(dims, locations, rotation_y, pitch, roll)
        # IMPORTANT NOTE! all samples provide the location of the object center.
        # zed samples, for some reason, define the y_corners at [0,-h], and cognata needs a definition of [-h/2,h/2]

        # to avoid drawing issues where bbox is partially unseen, we clip the depth
        for vertex in box_3d:
            # print(vertex)
            if vertex[2] < 0:
                vertex[2] = 0.001
            # print(vertex)

        # FOR IMAGRY SAMPLES:
        if sample_origin == 'zed':
            box_2d = fov_project_to_image_without_calib(box_3d, image_fov, input_w = new_size[1], input_h = new_size[0])
        elif sample_origin == 'sheba':
            box_2d = fov_project_to_image_without_calib(box_3d, np.deg2rad(image_fov), input_w = new_size[1], input_h = new_size[0])            
        else:
            # box_2d = fov_project_to_image_without_calib(box_3d, image_fov,
            #                                             input_w = new_size[1], input_h = new_size[0])
            box_2d = project_to_image(box_3d, calib.transpose(1,0))

        # if cat_id != 8:
        #     continue

        # cv2.putText(image, f'{box_2d[0][0]:.2f}', TL_values, font, 0.4,(0,0,255),1,cv2.LINE_AA)
        # print(f'\n{box_3d = }')
        # DRAW PROJECTED BOX
        # print(f'\n{box_2d = }')
        cv2.rectangle(image,(int(coco_bbox[0]),int(coco_bbox[1])),(int(coco_bbox[0])+ int(coco_bbox[2]),int(coco_bbox[1]) + int(coco_bbox[3])),(20,255,255), 2)
        image = draw_box_3d(image, box_2d)

    cv2.putText(image,'ANNOTATION',(50,50), font, 0.5,(0,0,255),1,cv2.LINE_AA)


    _, bird_view = exctract_3d_from_info(image, bird_view, 0, image_annotations, 1, np.shape(image)[0], np.shape(image)[1],
                                         src_img_file = None, trip = None, monodepth_array = None)
    # cv2.imshow('BEV',bird_view)
    # cv2.waitKey()

    bird_view_resized = imutils.resize(bird_view, height=np.shape(image)[0])
    catted = np.concatenate((image, bird_view_resized),axis=1)
    cv2.imshow('CATTED',catted)
    cv2.waitKey()

    name,ext = os.path.splitext(os.path.basename(image_name))
    cv2.imwrite(os.path.join(output_path ,name)+'_annotation'+ext, catted)
    # cv2.imwrite(os.path.join(output_path ,name)+'_annotation'+ext, image)



# a = 5
