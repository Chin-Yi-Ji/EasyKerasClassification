#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd 
import os
import glob


# In[2]:


#建立 dataset
def MakeDataSet(Path_list, Y_list, BATCH = 2, SIZE = 512, AUGMENT = False, SHUFFLE = False):
    
    Path = tf.data.Dataset.from_tensor_slices(Path_list)
    Y_list = tf.data.Dataset.from_tensor_slices(Y_list)
    
    #調用MakeFun
    _Resizeimages = MakeFun(SIZE,AUGMENT)
    
    #調用MakeFun 中的_ReadResizeImage 和 mask_resize
    Path = Path.map(_Resizeimages, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    Y = Y_list
    DataSet = tf.data.Dataset.zip((Path, Y))
    
    #data shuffle
    if SHUFFLE is True:
        DataSet = DataSet.shuffle(len(Path_list), reshuffle_each_iteration=False)
    DataSet = DataSet.batch(BATCH, drop_remainder = True)
    DataSet = DataSet.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return DataSet    


def MakeInferSet(Path_list,  BATCH = 2, SIZE = 512, AUGMENT = False, SHUFFLE = False):
    
    Path = tf.data.Dataset.from_tensor_slices(Path_list)
    
    #調用MakeFun
    _Resizeimages = MakeFun(SIZE,AUGMENT)
    
    #調用MakeFun 中的_ReadResizeImage 和 mask_resize
    Path = Path.map(_Resizeimages, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    DataSet = tf.data.Dataset.zip((Path, Y))
    
    #data shuffle
    if SHUFFLE is True:
        DataSet = DataSet.shuffle(len(Path_list), reshuffle_each_iteration=True)
    DataSet = DataSet.batch(BATCH, drop_remainder = True)
    DataSet = DataSet.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return DataSet 



#MakeFun 函數
def MakeFun(SIZE,AUGMENT):
    
    @tf.function
    def _Resizeimages(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        #image = tf.squeeze(image,axis = 0)
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        if AUGMENT is True:   
            image = tf.cond(tf.random.uniform([],0,1)>0.5, lambda: tf.image.random_hue(image,0.5), lambda: image)
            image = tf.cond(tf.random.uniform([],0,1)>0.5, lambda: tf.image.random_saturation(image,0,1), lambda: image)
            image = tf.cond(tf.random.uniform([],0,1)>0.5, lambda: tf.image.random_contrast(image,0.4,1), lambda: image)
            image = tf.cond(tf.random.uniform([],0,1)>0.5, lambda: tf.image.random_brightness(image,0.5), lambda: image)  
        
        
        # resize_image
        image = tf.image.resize(image, [SIZE,SIZE], method = 'bilinear')
        
        image = tf.cond( tf.math.equal(tf.shape(image)[-1],1), lambda: tf.concat((image,image,image),axis=-1) , lambda: image)
        
        return image    

   
    return _Resizeimages

