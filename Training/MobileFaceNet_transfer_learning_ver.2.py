# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:54:33 2019

@author: TMaysGGS
"""

'''Updated on 07/31/2019 11:14'''
'''Problem
There is an intermediate layer that has the shape (m, m), which means a 26928 * 26928 matrix. 
Each element takes 32 bit so that the total space needed is more than 21G, which way too much exceeds 
the GPU memory. 
'''
'''Loading the pre-trained model'''
import math
import os
import keras
import random
import cv2
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, PReLU, Input, SeparableConv2D, DepthwiseConv2D, add, Flatten, Dense, Dropout
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras import initializers
from keras.utils import to_categorical
# import keras.backend.tensorflow_backend as KTF
# from keras.utils import plot_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 如需多张卡设置为：'1, 2, 3'，使用CPU设置为：''
'''Set if the GPU memory needs to be restricted
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
session = tf.Session(config = config)

KTF.set_session(session)
'''
BATCH_SIZE = 512 
old_m = 1416143
m = 6151666 
DATA_SPLIT = 0.008
OLD_NUM_LABELS = 6181
NUM_LABELS = 26928 
TOTAL_EPOCHS = 10000 
IMG_DIR = '/data/daiwei/dataset' 

'''Importing the data set (not appropriate when the data set is too large)
from keras.preprocessing.image import ImageDataGenerator

train_path = '/data/daiwei/dataset/'

train_datagen = ImageDataGenerator(rescale = 1. / 255, validation_split = DATA_SPLIT)

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
'''

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
    label = Input((OLD_NUM_LABELS, ))

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
    
    M = ArcFaceLossLayer(class_num = OLD_NUM_LABELS)([M, label])
    Z_L = Dense(OLD_NUM_LABELS, activation = 'softmax')(M) 
    
    model = Model(inputs = [X, label], outputs = Z_L, name = 'mobile_face_net')
    
    return model

model = mobile_face_net()

# model.summary()
# model.layers

'''Loading the model & re-defining'''
print("Reading the pre-trained model") 
model.load_weights("/home/daiwei/Python_Coding/MobileFaceNet/tl_model1906100200.hdf5")
# model.load_weights("E:\\Python_Coding\\MobileFaceNet\\tl_model_1905270955.hdf5")
print("Reading done. ")
model.summary()
# model.layers

# Re-define the model
model.layers.pop() # Remove the last FC layer
model.layers.pop() # Remove the ArcFace Loss Layer
model.layers.pop() # Remove the Label Input Layer
model.summary()

model.layers[-1].outbound_nodes = []
model.outputs = [model.layers[-1].output] # Reset the output
output = model.get_layer(model.layers[-1].name).output
model.input
# The model used for prediction
pred_model = Model(model.input[0], output)
pred_model.summary()
# pred_model.save('pred_model.h5')

# Custom the model for continue training
label = Input((NUM_LABELS, ))
M = pred_model.output
M = ArcFaceLossLayer(class_num = NUM_LABELS)([M, label])
Z_L = Dense(NUM_LABELS, activation = 'softmax')(M)

customed_model = Model(inputs = [pred_model.input, label], outputs = Z_L, name = 'mobile_face_net_transfered')
customed_model.summary()
# customed_model.layers
# plot_model(customed_model, to_file='customed_model.png')

'''Setting configurations for training the Model''' 
customed_model.compile(optimizer = Adam(lr = 0.01, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Temporarily increase the learing rate to 0.01

# Save the model after every epoch
from keras.callbacks import ModelCheckpoint 
check_pointer = ModelCheckpoint(filepath = 'MobileFaceNet.hdf5', verbose = 1, save_best_only = True)

# Interrupt the training when the validation loss is not decreasing
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 1000)

# Record the loss history
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []
        
    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

# Stream each epoch results into a .csv file
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('training_log.csv', separator = ',', append = True)
# append = True append if file exists (useful for continuing training)
# append = False overwrite existing file

# Reduce learning rate when a metric has stopped improving
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 20, min_lr = 0)

'''Importing the data & training the model'''
'''
hist = customed_model.fit_generator(
        train_generator,
        steps_per_epoch = (m * (1 - DATA_SPLIT)) // BATCH_SIZE,
        epochs = 10000,
        callbacks = [check_pointer, early_stopping, history, csv_logger, reduce_lr], 
        validation_data = validate_generator, 
        validation_steps = (m * DATA_SPLIT) // BATCH_SIZE)
'''

data_list = []
for label_directory in os.listdir(IMG_DIR): 
    
    for img_name in os.listdir(os.path.join(IMG_DIR, label_directory)): 
        
        data_list.append([os.path.join(IMG_DIR, label_directory, img_name), int(label_directory)])

for i in range(TOTAL_EPOCHS): 
    
    random.shuffle(data_list) 
    num_of_iters = len(data_list) // BATCH_SIZE 
    for j in range(num_of_iters): 
        
        training_data_list = data_list[BATCH_SIZE * j: BATCH_SIZE * (j + 1)] 
        
        X_batch_list = []
        Y_batch_list = []
        for info in training_data_list: 
            
            img = cv2.imread(info[0]) 
            if img.shape != (112, 112, 3): 
                img = cv2.resize(img, (112, 112), interpolation = cv2.INTER_LINEAR) 
            # assert(img.shape == (112, 112, 3)) 
            X_batch_list.append((img - 127.5) * 0.00784313725490196)
            Y_batch_list.append(info[1]) 
        X_batch = np.array(X_batch_list) 
        Y_batch = np.array(Y_batch_list) 
        Y_batch = to_categorical(Y_batch, num_classes = NUM_LABELS)
        print("X_batch shape: " + str(X_batch.shape)) 
        print("Y_batch shape: " + str(Y_batch.shape)) 
            
        hist = customed_model.fit(
                [X_batch, Y_batch], Y_batch, 
                epochs = 1, 
                callbacks = [check_pointer, history, csv_logger, reduce_lr]) 
        
        print(hist.history) 
