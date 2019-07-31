# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:47:23 2019

@author: TMaysGGS
"""

import os 
import pickle as pkl 

IMG_DIR = '/data/daiwei/dataset' 

data_list = [] 
for directory in os.listdir(IMG_DIR): 
    for img_name in os.listdir(os.path.join(IMG_DIR, directory)): 
        img_path = os.path.join(IMG_DIR, directory, img_name) 
        data_list.append([img_path, directory]) 

save_path = '/data/daiwei/processed_data/MobileFaceNet/data_list.pkl'
file = open(save_path, 'wb+')
pkl.dump(data_list, file)
file.close()
