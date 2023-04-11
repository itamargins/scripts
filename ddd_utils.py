from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import copy 
from scipy.spatial import distance_matrix

from test_constants import *
# from opts import Res

# This is temporary
h_fov = np.deg2rad(85)
focal = 1/np.tan(h_fov/2)
norm_focal_length = focal/2

def get_tranformation_factor(depth):
    max_depth_threshold = 60
    min_tranformation_factor = 1.2 # The factor should be in range(1.2,1.7)
    return min_tranformation_factor + (min(depth, max_depth_threshold)/(2*max_depth_threshold))


def transform_poly_2d(poly_points, to_hpr, to_pos, t_mat=None):
    #rotate objects on left/right cams
    MM_HEIGHT_HALF = MM_HEIGHT / 2
    poly_points[:, 1] -= MM_HEIGHT_HALF

    to_pos /= pixel_length_in_mm
   
    poly_points = Transform_2D_opt(poly_points, to_pos, to_hpr)
    poly_points[:, 1] += MM_HEIGHT_HALF

    #cast back to cv2 poly function input
    poly_points = poly_points.astype(int)

    return poly_points

def Transform_2D_opt(xy, pos, yaw): 

    sin_x = np.sin(yaw) * xy[:, 0]
    cos_x = np.cos(yaw) * xy[:, 0]
    sin_y = np.sin(yaw) * xy[:, 1]
    cos_y = np.cos(yaw) * xy[:, 1]

    xy[:, 0] = cos_x - sin_y
    xy[:, 1] = sin_x + cos_y

    xy[:, 0] += pos[0]
    xy[:, 1] += pos[1]

    return xy

def bb_intersection_over_union(_boxA, _boxB,w,h):
    # Given box = [x0,y0,w,h]
    boxA = copy.deepcopy(_boxA)
    boxB = copy.deepcopy(_boxB)
    boxA[2] +=  boxA[0]
    boxA[3] +=  boxA[1]
    boxB[2] +=  boxB[0]
    boxB[3] +=  boxB[1]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA+ 1/w) * max(0, yB - yA + 1/h)
    boxAArea = (boxA[2] - boxA[0] + 1/w) * (boxA[3] - boxA[1] + 1/h)
    boxBArea = (boxB[2] - boxB[0] + 1/w) * (boxB[3] - boxB[1] + 1/h)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_hpr(trip):
    if trip in data_0:
        hpr = np.array([0, 0, 0], dtype=float)
    elif trip in data_40:
        hpr = np.array([np.deg2rad(40), 0, 0], dtype=float)
    elif trip in data_40:
        hpr = np.array([np.deg2rad(65), 0, 0], dtype=float)
    else:
        hpr = np.array([np.deg2rad(75), 0, 0], dtype=float)
    return hpr

# Where a = line point 1; b = line point 2; c = point to check against.
# If the formula is equal to 0, the points are colinear.
# If the line is horizontal, then this returns true if the point is above the line.
def isLeft(a,b,c):
    return ((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])) > 0


def rbox_line_intersection(box,line):
    points = [line_intersect(box[0][0], box[0][1], box[1][0], box[1][1], line[0][0], line[0][1], line[1][0], line[1][1]),
    line_intersect(box[1][0], box[1][1], box[2][0], box[2][1], line[0][0], line[0][1], line[1][0], line[1][1]),
    line_intersect(box[2][0], box[2][1], box[3][0], box[3][1], line[0][0], line[0][1], line[1][0], line[1][1]),
    line_intersect(box[3][0], box[3][1], box[0][0], box[0][1], line[0][0], line[0][1], line[1][0], line[1][1])
    ]
    points = [x for x in points if x is not None]
    return points

def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
 
    return x, y


# Add version for all function that need h_fov as variable
def ddd_asu2locrot(yolo_box, alpha, dim, depth, azimuth_shift):
  #yolo_box: xc,yc,w,h
  norm_focal_length = focal/2
  xcr = yolo_box[0]
  azimuth_2d = np.arctan((xcr - 0.5)/norm_focal_length) 
  
  rotation_y = alpha + azimuth_2d

  Y = get_Ycam(yolo_box,depth)

  xcr_3d = norm_focal_length * np.tan(azimuth_2d) + 0.5

  x_mm = (xcr_3d - 0.5)*(depth/norm_focal_length)

  locations = np.array([x_mm,Y,depth], dtype=np.float32)

  return locations, rotation_y

def compute_azimuth(xcr, fov):
  _focal = 1/np.tan(fov/2)
  norm_focal_length = _focal/2
  azimuth = np.arctan((xcr - 0.5)/norm_focal_length)
  return azimuth

def cx_from_azimuth(azimuth, fov):        
  _focal = 1/np.tan(fov/2)
  norm_focal_length = _focal/2
  deg_half_fov = np.rad2deg(fov/2)
  deg_az = np.rad2deg(azimuth)
  if abs(azimuth) > fov/2:
      multiples = deg_az//deg_half_fov
      azimuth = np.deg2rad(abs(deg_az)%deg_half_fov)
      if azimuth > 0:
          xcr = np.tan(azimuth)*norm_focal_length + 0.5
      else:
          azimuth = -azimuth
          xcr = np.tan(azimuth)*norm_focal_length + 0.5
      xcr += multiples
  else:
      xcr = np.tan(azimuth)*norm_focal_length + 0.5
  return xcr


def fov_ddd_asu2locrot(yolo_box, alpha, dim, depth, fov, azimuth_shift, cls_ind, batch_index, opt, x_location = None, y_location = None):
  # The field of view is changes
  if opt.inference_batch_config.res_palette[batch_index] == Res.HD1080:
    fov = np.deg2rad(69)
  _focal = 1/np.tan(fov/2)
  #yolo_box: xc,yc,w,h
  norm_focal_length = _focal/2
  xcr = yolo_box[0]
  azimuth_2d = np.arctan((xcr - 0.5)/norm_focal_length)
  rotation_y = alpha + azimuth_2d

  if rotation_y > np.pi:
      rotation_y -= 2 * np.pi
  if rotation_y < -np.pi:
      rotation_y += 2 * np.pi
  
  if opt.inference_batch_config.res_palette[batch_index] == Res.HD1080:
    depth = depth * get_tranformation_factor(depth)
  Y = fov_get_Ycam(yolo_box, depth, fov)

  xcr_3d = norm_focal_length * np.tan(azimuth_2d) + 0.5

  x_mm = (xcr_3d - 0.5)*(depth/norm_focal_length)

  locations = np.array([x_mm,Y,depth], dtype=np.float32)
  
  if abs(depth) < 1 or abs(xcr_3d) > 3.5 or (abs(azimuth_2d) > np.deg2rad(50)):
    if x_location and y_location:
      #print('x_location', x_location, 'y_location', y_location)
      locations[0] = np.sign(x_location)*np.sqrt(abs(x_location)) if abs(x_location) > 6 else x_location
    else:
      locations[0] = np.sign(azimuth_2d) * 4
    locations[1] = 1
  
  if cls_ind-1 in [5,6,7] and x_location:
    print(f'locations[0] = {locations[0]}, x_location = {x_location}')
    locations[0] = x_location
  
  # if cls_ind-1 == 7:
  #   print('abs(depth)', abs(depth), 'abs(xcr_3d)', abs(xcr_3d), 'azimuth_2d', np.rad2deg(azimuth_2d), 'locations', locations)
    #print('after: azimuth_2d', np.rad2deg(azimuth_2d),'locations', locations, 'xcr_3d', xcr_3d, 'rotation_y', np.rad2deg(rotation_y), 'cls_ind-1', cls_ind-1,'\n')
      
  return locations, rotation_y

def fov_get_Ycam(box, depth, fov, im_width=1280, im_height=720):
  _focal = 1/np.tan(fov/2)
  aspect_ratio_f =  (im_height/im_width)*2
  y = box[1] - (box[3]/2)
  y_ = (0.5-y) * aspect_ratio_f

  Y = (y_ * depth)/_focal 
  return Y

def _distance(a, b):
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return (dx ** 2 + dy ** 2) ** 0.5


def _get_split_point(a, b, dist):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dx == 0:
        x = a[0]
        y = min(a[1],b[1]) + dist
        return int(x), int(y)
    if dx < 0:
        a, b = b, a
        dx = b[0] - a[0]
        dy = b[1] - a[1]
    m = dy / dx
    c = a[1] - (m * a[0])

    x = a[0] + (dist**2 / (1 + m**2))**0.5
    y = m * x + c
    # formula has two solutions, so check the value to be returned is
    # on the line a b.
    if not (a[0] <= x <= b[0]) and (a[1] <= y <= b[1]):
        x = a[0] - (dist**2 / (1 + m**2))**0.5
        y = m * x + c

    return int(x), int(y)

def find_intersection(poly):
  for i in range(poly.shape[0]):
    for j in range(i+1,poly.shape[0]):
      a = poly[i]
      b = poly[j]
      dist = _distance(a,b)
      for counter in range(10):
        p = _get_split_point(a, b, counter*dist/9)
        if p[0] > p0[0] and p[0] < p1[0] and p[1] > p0[1] and p[1] < p1[1]:
          return True
  return False
  
def gen_bird_view_image():
  bird_view = np.ones((mm_height, mm_width, 3), dtype=np.uint8) * 230
  base = (left, big_car_center)
  cv2.rectangle(bird_view, p0, p1,(0,0,0), cv2.FILLED)
  cv2.line(bird_view, base, left_h_fov_point,(0,0,0))
  cv2.line(bird_view, base, right_h_fov_point,(0,0,0))
  for rad in np.arange(d*meter_to_pixel,mm_width,d*meter_to_pixel):
      bird_view = cv2.circle(bird_view, (0+left,big_car_center), int(round(rad)), (0,128,128),1, lineType=cv2.LINE_AA)
      cv2.putText(bird_view, str(int(round(rad*1/meter_to_pixel))), (0+left+int(round(rad)), big_car_center), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
  return bird_view

# 0, pi/2, -pi/2, pi
def get_rotationy_direction(ry):
    PI = np.pi
    if ry > -PI/4 and ry <= PI/4:
        return 0
    if ry > PI/4 and ry <= 3*PI/4:
        return np.pi/2
    if ry > 3*PI/4 or ry < -3*PI/4:
        return np.pi
    return -np.pi/2


def fov_project_to_image_without_calib(box_3d, fov, input_w = 1280, input_h = 720):
  _focal = 1/np.tan(fov/2)
  cx = input_w / 2
  cy = input_h  / 2
  projection_factor = _focal/box_3d[:,2]
  projection_factor[(box_3d[:,2]<0.1)] = _focal/0.1
  #projecting lidar points to image plane 
  xis = box_3d[:,0] * projection_factor 
  xis = xis * cx + cx
  yis = box_3d[:,1] * projection_factor 
  yis = cy - cx * yis
  return np.array(list(zip(xis, yis)))

def draw_box_3d(image, corners, c=(0, 0, 255)):
    lines = []
    face_idx = [[0,3,4,1],
                [2,1,6,3],
                [5,6,1,4],
                [7,4,3,6]]
    colorsss = [(255,0,0), (255,0,0), (255,0,0), (0,255,0), (0,0,255), (0,0,255), \
                    (0,255,0), (0,255,0), (0,255,0), (255,0,0), (0,0,255), (0,0,255)]
    points = [0,2,5,7]
    for ix,xy in enumerate(list(zip(corners[points,0], corners[points,1]))):
        # print(ix, xy)
        x,y = xy
        # print(face_idx[ix])
        for fix, face in enumerate(face_idx[ix]):
          if fix == 0:
              continue
          x2,y2 = corners[face, 0], corners[face, 1]
          line = [(x,y), (x2,y2)]
          lines.append(line)
    for l,line in enumerate(lines):
        c = colorsss[l]
        line_1 = (int(line[0][0]) , int(line[0][1]))
        line_2 = (int(line[1][0]), int(line[1][1]))
        cv2.line(image, line_1, line_2, c, lineType=cv2.LINE_AA)
    return image

# For MKZ it is good.
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

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

# def project_3d_to_bird(pt, out_size,):
#   world_size = 64
#   pt[0] += world_size / 2
#   pt[1] = world_size - pt[1]
#   pt = pt * out_size / world_size
#   return pt.astype(np.int32)


def project_3d_to_bird(pt, world_size_x=64, world_size_y=64, out_size_x=192, out_size_y=192, ratio=192/64, h_world_size_x=32, rotate=False):
  pt[0] += h_world_size_x
  pt[1] = world_size_y - pt[1]
  pt[0] = pt[0] * ratio
  pt[1] = pt[1] * ratio
  # pt = pt.astype(np.int32)
  if rotate:
    pt = np.array([out_size_y-pt[1], pt[0]])
  return pt


def bird_view_box(rect, mm, world_cfg=world_cfg, nb_points = 4):
  for k in range(nb_points):
    if mm:
      rect[k] = project_3d_to_bird(rect[k], 
                                    world_cfg['world_size_x'],
                                    world_cfg['world_size_y'], 
                                    world_cfg['out_size_x'], 
                                    world_cfg['out_size_y'],
                                    world_cfg['world_ratio'],
                                    world_cfg['h_world_size_x'],
                                     rotate=True)
    else:
      rect[k] = project_3d_to_bird(rect[k])
  return rect

def add_bird_view(bird_view, rect, mm=False, world_cfg=world_cfg):
  '''get_poly_points() + draw_box_bird_view()
  '''
  lc = (250, 152, 12)
  front_color = (0,255,0)
  #bird_view_rect
  rect = bird_view_box(rect, mm, world_cfg=world_cfg)
  poly_points = [rect.reshape(-1, 1, 2).astype(np.int32)]
  cv2.polylines(bird_view, poly_points, True, lc, 2, lineType=cv2.LINE_AA)
  cx,cy = np.mean(rect[:,0]), np.mean(rect[:,1])
  bird_view = cv2.circle(bird_view, (cx,cy), 1, (0,0,255),1)
  
  for e in [[0, 1]]:
    if e == [0, 1]:
      t = 2
      cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
            (rect[e[1]][0], rect[e[1]][1]), front_color, t,
            lineType=cv2.LINE_AA)
    else:
      t = 1
      cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
            (rect[e[1]][0], rect[e[1]][1]), lc, t,
            lineType=cv2.LINE_AA)
  return bird_view, poly_points


def draw_box_bird_view(bird_view, rect, poly_points, j, t=0, rotated_box_min_point = None):
  if j == 0:
    lc = (0, 255, 255)
  elif j == 1:
     lc = (255, 0, 255)
  elif j == 2:
    lc = (255, 0, 0)
  elif j == 3:
    lc = (0, 0, 255)
  front_color = (0,0,0)
  t = t if t != 0 else 1
  if rotated_box_min_point is not None:
    # Calculate the minimu depth point
    rotated_box_min_point  = rotated_box_min_point[0]
    bird_view = cv2.circle(bird_view, (int(round(rotated_box_min_point[0])),int(round(rotated_box_min_point[1]))), 1, (0,0,255), t+1)
  poly_points = np.int64([poly_points])
  cv2.polylines(bird_view, poly_points, True, lc, t, lineType=cv2.LINE_AA)
  cx,cy = np.mean(rect[:,0]), np.mean(rect[:,1])
  try:
    bird_view = cv2.circle(bird_view, (int(round(cx)),int(round(cy))), 1, (255,0,0), t)
  except:
    pass
  p_1 = (int(round(np.round(rect[0][0]))), int(round(np.round(rect[0][1]))))
  p_2 = (int(round(np.round(rect[1][0]))), int(round(np.round(rect[1][1]))))
  cv2.line(bird_view, p_1, p_2, front_color, t, lineType=cv2.LINE_AA)
  return bird_view


def get_poly_points(rect, mm=False, world_cfg=world_cfg, nb_points = 4):
  #bird_view_rect
  rect = bird_view_box(rect, mm, world_cfg=world_cfg, nb_points = nb_points)
  poly_points = rect.reshape(-1, 1, 2)#.astype(np.int32)
  return poly_points

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot(center, alpha, dim, depth, calib):
  # single image
  locations = unproject_2d_to_3d(center, depth, calib)
  locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y
  
def get_Ycam(box, depth, im_width=1280, im_height=720):
    aspect_ratio_f =  (im_height/im_width)*2
    y = box[1] - (box[3]/2)
    y_ = (0.5-y) * aspect_ratio_f

    Y = (y_ * depth)/focal 
    return Y

# x,y,w,h
def check_2d_box_update(bbox):
  width = 1280
  height = 720
  x0 = bbox[0]
  y0 = bbox[1]
  x1 = bbox[0] + bbox[2]
  y1 = bbox[1] + bbox[3]
  status = True
  if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
    #print('bbox before', bbox)
    print('before: negative values for x0, y0, x1, y1', x0, y0, x1, y1)
    status = False
    x0 = 0 if x0 < 0 else x0
    y0 = 0 if y0 < 0 else y0
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    print('after: big value for x0, y0, x1, y1', x0, y0, x1, y1)
  elif x0 > width or x1 > width or y0 > height or y1 > height:
    #print('bbox before', bbox)
    print('before: big value for x0, y0, x1, y1', x0, y0, x1, y1)
    status = False
    x0 = width-1 if x0 > width-1 else x0
    y0 = height-1 if y0 > height-1 else y0
    x1 = width-1 if x1 > width-1 else x1
    y1 = height-1 if y1 > height-1 else y1
    print('after: big value for x0, y0, x1, y1', x0, y0, x1, y1)
  bbox = [x0, y0, x1-x0, y1-y0]
  return status, bbox
  
def to_coco_box(bbox, x_factor = 1, y_factor = 1):
    x0,y0,w,h = bbox
    x0 = int(x0*x_factor)
    w = int(w*x_factor)
    y0 = int(y0*y_factor)
    h = int(h*y_factor)
    return [x0,y0,w,h]

# def project_3d_bbox(location, dim, rotation_y, calib):
#   box_3d = compute_box_3d(dim, location, rotation_y)
#   box_2d = project_to_image(box_3d, calib)
#   return box_2d



def points_within_fov(left_fov_vec, right_fov_vec, X):
    b = 0
    origin = (0, 0)
    
    m_left = (left_fov_vec[1] - origin[1]) / (left_fov_vec[0] - origin[0])
    yf_left = lambda x: m_left * x + b
    y_left = yf_left(X[:, 0])
    above_line_left = np.flatnonzero(X[:, 1] >= y_left) #v=1
    below_line_left = np.flatnonzero(X[:, 1] < y_left) #v=-1
    
    m_right = (right_fov_vec[1] - origin[1]) / (right_fov_vec[0] - origin[0])
    yf_right = lambda x: m_right * x + b
    y_right = yf_right(X[:, 0])
    above_line_right = np.flatnonzero(X[:, 1] >= y_right) #v=1
    below_line_right = np.flatnonzero(X[:, 1] < y_right) #v=-1
    
    ps_above_line_left = X[above_line_left]
    ps_above_line_right = X[above_line_right]
    
    union_ixs = np.flatnonzero(np.in1d(above_line_left, above_line_right))
    union_ixs = above_line_left[union_ixs]
    ps_with_fov = X[union_ixs]
    
    return ps_with_fov

def get_closest_y_point(X):
    '''
    fov_slice: 'left' or 'right' string
    '''
    X = np.insert(X, len(X), X[0], axis=0)
    start_x = X[:-1, 0]
    end_x = X[1:, 0]
    start_x = np.insert(start_x, len(X)-1, X[0,0])
    end_x = np.insert(end_x, len(X)-1, X[1,0])

    start_y = X[:-1, 1]
    end_y = X[1:, 1]
    start_y = np.insert(start_y, len(X)-1, X[0,1])
    end_y = np.insert(end_y, len(X)-1, X[1,1])

    # print(start_x)
    # print(end_x)

    count_l = np.abs((start_x - end_x) / d_pos).astype(int)
    max_count_l = np.max(count_l)

    # print(count_l)

    NXs = []
    for i in range(len(count_l)-1):
        linx = np.linspace(start_x[i], end_x[i], max_count_l)
        liny = np.linspace(start_y[i], end_y[i], max_count_l)
        NX = np.concatenate([np.expand_dims(linx, -1), np.expand_dims(liny, -1)], axis=1)
        NXs.append(NX)
    NXs = np.concatenate(NXs)
    
    X_within = points_within_fov(left_h_fov_line, right_h_fov_line, NXs)

    min_y_ix = np.argmin(X_within[:, 1])
    point_w_closest_y = X_within[min_y_ix]
    
    return point_w_closest_y #, X_within


def fix_rect_by_2d_box(dims, rect, bbox, is_side_cam, image_shape = (416,736)):
  bbox[0], bbox[2] = bbox[0]/image_shape[1], bbox[2]/image_shape[1]
  max_dim = max(dims[1], dims[2])
  min_x_rect = rect[np.argmin(rect[:,0])].copy()
  max_x_rect = rect[np.argmax(rect[:,0])].copy()

  ratio_error = (np.max(rect[:,1])- np.min(rect[:,1]))/max_dim
  mean_depth = np.mean(rect[(rect[:,1] > 0),1]) if ratio_error > 0.8 else np.max(rect[:,1])

  min_x_rect[1] = max(mean_depth, min_x_rect[1]) if min_x_rect[1] < 3 else min_x_rect[1]
  max_x_rect[1] = max(mean_depth, max_x_rect[1]) if max_x_rect[1] < 3 else max_x_rect[1]

  min_x = (bbox[0] - 0.5) * (min_x_rect[1]/norm_focal_length)
  max_x = (bbox[2] - 0.5) * (max_x_rect[1]/norm_focal_length)

  for i in range(len(rect)):
    rect[i,0] = min(max(rect[i,0], min_x), max_x)
  

def padd_dims(rect, dims):
  distance_m = distance_matrix(rect, rect)
  small_dim = min(dims[1],dims[2])/3
  if distance_m[0,1] < 2 and distance_m[2,3] < 2 and distance_m[1,2] > 5 and distance_m[0,3] > 5:
    rect[0,0] -= small_dim
    rect[1,0] += small_dim
    rect[2,0] += small_dim
    rect[3,0] -= small_dim


def get_range(dep, az):
  return dep/np.cos(az)


if __name__ == '__main__':
  calib = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  alpha = -0.20
  tl = np.array([712.40, 143.00], dtype=np.float32)
  br = np.array([810.73, 307.92], dtype=np.float32)
  ct = (tl + br) / 2
  rotation_y = 0.01
  # print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
  # print('rotation_y', rotation_y)
