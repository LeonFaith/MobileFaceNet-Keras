# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:44:17 2019

@author: TMaysGGS
"""

'''Last updated on 07/23/2019 16:48''' 
import os 
import cv2 
import tensorflow as tf 

# Transfer data types 
def _int64_feature(value): 
    if not isinstance(value, list): 
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value)) 

def _float_feature(value): 
    if not isinstance(value, list): 
        value = [value] 
    return tf.train.Feature(float_list = tf.train.FloatList(value = value)) 

def _bytes_feature(value): 
    if not isinstance(value, list): 
        value = [value]
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = value)) 

SRC_PATH = '/data/daiwei/dataset' 
TFRECORD_PATH = '/data/daiwei/processed_data/MobileFaceNet/training.tfrecords' 
MAPFILE_PATH = '/data/daiwei/processed_data/MobileFaceNet/traning_classmap.txt' 

labels = os.listdir(SRC_PATH) 
class_num = len(labels) 

writer = tf.python_io.TFRecordWriter(TFRECORD_PATH) 
class_map = {} 
for idx, label in enumerate(labels): 
    class_path = os.path.join(SRC_PATH, label) 
    class_map[idx] = label 
    for img_name in os.listdir(class_path): 
        
        # Read the image & preprocess 
        img_path = os.path.join(class_path, img_name) 
        
        img = cv2.imread(img_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        if img.shape[0] != 112 or img.shape[1] != 112: 
            img = cv2.resize(img, (112, 112), interpolation = cv2.INTER_LINEAR) 
        
        # Transfer the image into binary mode 
        img_str = img.tostring() 
        example = tf.train.Example(features = tf.train.Features(feature = {
                'image': _bytes_feature(img_str), 
                'label': _int64_feature(int(label))
                })) 
        writer.write(example.SerializeToString()) 
writer.close() 

txtfile = open(MAPFILE_PATH, 'w+') 
for key in class_map.keys(): 
    txtfile.writelines(str(key) + ":" + class_map[key] + "\n") 
txtfile.close() 
