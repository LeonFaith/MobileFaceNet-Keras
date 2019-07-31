# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:06:01 2019

@author: TMaysGGS
"""

'''Renaming the folders with the order starting from 0'''
import os
import shutil

def batch_rename_folders(folder_path):
    
    i = 0
    for directory in os.listdir(folder_path):
        
        print(directory)
        
        shutil.move(folder_path + '/' + directory, folder_path + '/xyz' + str(i))
        i = i + 1
    
    i = 0
    for directory in os.listdir(folder_path):
        
        print(directory)
        
        shutil.move(folder_path + '/' + directory, folder_path + '/' + str(i))
        i = i + 1
    
    return i

folder_path = '/data/daiwei/dataset'
i = batch_rename_folders(folder_path)
print(str(i))
