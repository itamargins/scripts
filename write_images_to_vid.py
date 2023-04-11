import cv2
import numpy as np
import glob
import os

#####################################################################################
frame_size = (1280,720)
directory = '/home/itamar/Desktop/test/continuous_trip/predictions_model_2/training_IMAGRY_data/'
out = cv2.VideoWriter(directory+'prediction_video.avi',\
                       cv2.VideoWriter_fourcc(*'DIVX'), 6, frame_size)

#####################################################################################

for i,filename in enumerate(sorted(glob.glob(directory+'*.jpeg'))):
    if 'BEV' in filename:
        continue
    print(f'{i}: {filename}')
    img = cv2.imread(filename)
    out.write(img)

out.release()