# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:39:22 2019

@author: TMaysGGS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:29:04 2019

@author: TMaysGGS
"""

'''Image Preprocessing'''
import os
import random
import cv2
import skimage
import numpy as np
import tensorflow as tf

import detect_face

os.environ['CUDA_VISIBLE_DEVICES']=''
input_path = '/data/face_data/glin_data/celebrity/' # path of raw image data set
output_path = '/data/daiwei/processed_data/cele/' # path of output image data set

def face_locating_and_resizing(img, saving_path):
    
    tf.reset_default_graph()
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 40 # minimum size of face
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bounding_boxes, points = detect_face.detect_face(temp, minsize, pnet, rnet, onet, threshold, factor)
    if bounding_boxes.shape[0] == 1 and bounding_boxes[0, 0] >= 0 and bounding_boxes[0, 1] >= 0 and bounding_boxes[0, 2] >= 0 and bounding_boxes[0, 3] >= 0:
        for b in bounding_boxes:
            draw = img[int(b[1]): int(b[3]), int(b[0]): int(b[2])]
        
        draw = cv2.resize(draw, (112, 112), interpolation = cv2.INTER_LANCZOS4)
        cv2.imwrite(saving_path, draw)

def data_clean_and_augmentation(input_path, output_path):
     
    nums_of_imgs = []
    paths_dir = []
    kernel = np.ones((5, 5), dtype = np.uint8)
    
    for directory in os.listdir(input_path): 
        
        # directory: label folder name
        print('\n' + directory +'\n')
        if os.path.isdir(input_path + directory) and not os.path.exists(output_path + directory):
            
            path_dir = os.listdir(input_path + directory) # The name list of the images in each label folder
            paths_dir.append(path_dir)
            
            num_of_imgs = len(path_dir) # The number of images in each label folder
            nums_of_imgs.append(num_of_imgs) # A list of the numbers of images in each label folder
            
            if num_of_imgs > 30:
                os.makedirs(output_path + directory)
                
            # For each label folder: 
            
            # (1) num > 350
            if num_of_imgs > 350:
                
                samples = random.sample(path_dir, 350) # Randomly pick 350 out from the overall set
                
                for name in samples:
                    
                    img = cv2.imread(input_path + directory + '/' + name)
                    
                    saving_path = output_path + directory + '/' + name
                    print(saving_path)
                    face_locating_and_resizing(img, saving_path)
                    
            # (2) 200 < num <= 350
            elif num_of_imgs > 200 and num_of_imgs <= 350:
                
                for name in path_dir:
                    
                    img = cv2.imread(input_path + directory + '/' + name)
                    
                    saving_path = output_path + directory + '/' + name
                    print(saving_path)
                    face_locating_and_resizing(img, saving_path)
         
            # (3) 90 < num <= 200
            elif num_of_imgs > 90 and num_of_imgs <= 200:
                
                for name in path_dir:
                    
                    img = cv2.imread(input_path + directory + '/' + name)
                    
                    saving_path_1 = output_path + directory + '/' + name
                    print(saving_path_1)
                    face_locating_and_resizing(img, saving_path_1)
                    
                    # Opening
                    temp_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                    
                    temp_img_name = 'Open_' + name
                    saving_path_2 = output_path + directory + '/' + temp_img_name
                    print(saving_path_2)
                    face_locating_and_resizing(temp_img, saving_path_2)
                    
            # (4) 30 < num <= 90
            elif num_of_imgs > 30 and num_of_imgs <= 90:
                
                for name in path_dir:
                    
                    img = cv2.imread(input_path + directory + '/' + name)
                    
                    saving_path_1 = output_path + directory + '/' + name
                    print(saving_path_1)
                    face_locating_and_resizing(img, saving_path_1)
                    
                    # Opening
                    temp_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                    
                    temp_img_name = 'Open_' + name
                    saving_path_2 = output_path + directory + '/' + temp_img_name
                    print(saving_path_2)
                    face_locating_and_resizing(temp_img, saving_path_2)
                    
                    # Add Gaussian noise
                    temp_img_2 = skimage.util.random_noise(img, mode = 'gaussian')
                    temp_img_2 = np.asarray(temp_img_2 * 255, dtype = np.uint8)
                    
                    temp_img_2_name = 'AddGaussian_' + name
                    saving_path_3 = output_path + directory + '/' + temp_img_2_name
                    print(saving_path_3)
                    face_locating_and_resizing(temp_img_2, saving_path_3)
                    
                    # Add Salt & Pepper noise 
                    temp_img_3 = skimage.util.random_noise(img, mode = 's&p')
                    temp_img_3 = np.asarray(temp_img_3 * 255, dtype = np.uint8)
                    
                    temp_img_3_name = 'AddSP_' + name
                    saving_path_4 = output_path + directory + '/' + temp_img_3_name
                    print(saving_path_4)
                    face_locating_and_resizing(temp_img_3, saving_path_4)
                    
            # (5) Drop the folder with num <= 30

data_clean_and_augmentation(input_path, output_path)

'''
if __name__ == '__main__':
	input_path = "./source/" # Source image folder path
	output_path = './result/' # Destination image folder path
	data_preprocessing_adjust_folders(input_path, output_path)
'''
