# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:58:15 2019

@author: TMaysGGS
"""

'''Importing the libraries'''
import math
# import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, PReLU, Input, SeparableConv2D, DepthwiseConv2D, add, Flatten, Dense, Dropout
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras import initializers

os.environ['CUDA_VISIBLE_DEVICES']='3'

BATCH_SIZE = 128
m = 72360

'''Importing the data set'''
from keras.preprocessing.image import ImageDataGenerator

train_path = '/data/dataset/'

train_datagen = ImageDataGenerator(rescale = 1. / 255, validation_split = 0.03)

def mobilefacenet_input_generator(generator, directory, subset):
    
    gen = generator.flow_from_directory(
            directory, 
            target_size = (112, 112), 
            color_mode = 'rgb', 
            batch_size = BATCH_SIZE, 
            class_mode = 'categorical', 
            subset = subset)
    
    while True: 
        
        X = gen.next()
        yield [X[0], X[1]], X[1] 

train_generator = mobilefacenet_input_generator(train_datagen, train_path, 'training')

validate_generator = mobilefacenet_input_generator(train_datagen, train_path, 'validation')

'''Building Block Functions'''
def conv_block(inputs, filters, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = Conv2D(filters, kernel_size, padding = "valid", strides = strides)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    A = PReLU()(Z)
    
    return A

def separable_conv_block(inputs, filters, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = SeparableConv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = "same")(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    A = PReLU()(Z)
    
    return A

def bottleneck(inputs, filters, kernel, t, s, r = False):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    
    Z1 = conv_block(inputs, tchannel, (1, 1), (1, 1))
    
    Z1 = DepthwiseConv2D(kernel_size = kernel, strides = s, padding = "same", depth_multiplier = 1)(Z1)
    Z1 = BatchNormalization(axis = channel_axis)(Z1)
    A1 = PReLU()(Z1)
    
    Z2 = Conv2D(filters, kernel_size = 1, strides = 1, padding = "same")(A1)
    Z2 = BatchNormalization(axis = channel_axis)(Z2)
    
    if r:
        Z2 = add([Z2, inputs])
    
    return Z2

def inverted_residual_block(inputs, filters, kernel, t, strides, n):
    
    Z = bottleneck(inputs, filters, kernel, t, strides)
    
    for i in range(1, n):
        Z = bottleneck(Z, filters, kernel, t, 1, True)
    
    return Z

def linear_GD_conv_block(inputs, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = DepthwiseConv2D(kernel_size = kernel_size, strides = strides, padding = "valid", depth_multiplier = 1)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    
    return Z

# Arc Face Loss Layer (Class)
class ArcFaceLossLayer(Layer):
    '''
    Arguments:
        inputs: the input embedding vectors
        class_num: number of classes
        s: scaler value (default as 64)
        m: the margin value (default as 0.5)
    Returns:
        the final calculated outputs
    '''
    def __init__(self, class_num, s = 64., m = 0.5, **kwargs):
        
        self.init = initializers.get('glorot_uniform') # Xavier uniform intializer
        self.class_num = class_num
        self.s = s
        self.m = m
        super(ArcFaceLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
        self.W = self.add_weight((input_shape[0][-1], self.class_num), initializer = self.init, name = '{}_W'.format(self.name))
        super(ArcFaceLossLayer, self).build(input_shape)
        
    def call(self, inputs, mask = None):
        
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)
        
        # features
        X = inputs[0] 
        # 1-D or one-hot label works as mask
        Y_mask = inputs[1] 
        # If Y_mask is not in one-hot form, transfer it to one-hot form.
        if Y_mask.shape[-1] == 1: 
            Y_mask = K.cast(Y_mask, tf.int32)
            Y_mask = K.reshape(K.one_hot(Y_mask, self.class_num), (-1, self.class_num))
        
        X_normed = K.l2_normalize(X, axis = 1) # L2 Normalized X
        self.W = K.l2_normalize(self.W, axis = 0) # L2 Normalized Weights
        
        # cos(theta + m)
        cos_theta = K.dot(X_normed, self.W)
        cos_theta2 = K.square(cos_theta)
        sin_theta2 = 1. - cos_theta2
        sin_theta = K.sqrt(sin_theta2 + K.epsilon())
        cos_tm = self.s * ((cos_theta * cos_m) - (sin_theta * sin_m))
        
        # This condition controls the theta + m should in range [0, pi]
        #   0 <= theta + m < = pi
        #   -m <= theta <= pi - m
        cond_v = cos_theta - threshold
        cond = K.cast(K.relu(cond_v), dtype = tf.bool)
        keep_val = self.s * (cos_theta - mm)
        cos_tm_temp = tf.where(cond, cos_tm, keep_val)
        
        # mask by label
        Y_mask =+ K.epsilon()
        inv_mask = 1. - Y_mask
        s_cos_theta = self.s * cos_theta
        
        output = K.softmax((s_cos_theta * inv_mask) + (cos_tm_temp * Y_mask))
        
        return output
    
    def compute_output_shape(self, input_shape):
        
        return input_shape[0], self.class_num

'''Building the MobileFaceNet Model'''
def mobile_face_net():
    
    X = Input(shape = (112, 112, 3))
    label = Input((320, ))

    M = conv_block(X, 64, 3, 2)

    M = separable_conv_block(M, 64, 3, 1)
    
    M = inverted_residual_block(M, 64, 3, t = 2, strides = 2, n = 5)
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1)
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 6)
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1)
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 2)
    
    M = conv_block(M, 512, 1, 1)
    
    M = linear_GD_conv_block(M, 7, 1) # kernel_size = 7 for 112 x 112; 4 for 64 x 64
    
    M = conv_block(M, 128, 1, 1)
    M = Dropout(rate = 0.1)(M)
    M = Flatten()(M)
    
    M = ArcFaceLossLayer(class_num = 320)([M, label])
    Z_L = Dense(320, activation = 'softmax')(M) 
    
    model = Model(inputs = [X, label], outputs = Z_L, name = 'mobile_face_net')
    
    return model

model = mobile_face_net()

model.summary()
model.layers

'''Training the Model'''
# Train on multiple GPUs
# from keras.utils import multi_gpu_model
# model = multi_gpu_model(model, gpus = 2)

model.compile(optimizer = Adam(lr = 0.001, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Save the model after every epoch
from keras.callbacks import ModelCheckpoint 
check_pointer = ModelCheckpoint(filepath = 'model.hdf5', verbose = 1, save_best_only = True)

# Interrupt the training when the validation loss is not decreasing
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10000)

# Record the loss history
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []
        
    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

# Stream each epoch results into a .csv file
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('training.csv', separator = ',', append = True)
# append = True append if file exists (useful for continuing training)
# append = False overwrite existing file

# Reduce learning rate when a metric has stopped improving
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 200, min_lr = 0)

hist = model.fit_generator(
        train_generator,
        steps_per_epoch = (m * 0.97) // BATCH_SIZE,
        epochs = 10000,
        callbacks = [check_pointer, early_stopping, history, csv_logger, reduce_lr],
        validation_data = validate_generator, 
        validation_steps = (m * 0.03) // BATCH_SIZE)

print(hist.history)
