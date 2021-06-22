#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_addons as tfa


def Optimizer(optimizer, learningrate):
    #tf.keras.optimizers
    if optimizer == 'Adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate = learningrate)
        
    elif optimizer == 'Adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate = learningrate)
        
    elif optimizer == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate = learningrate)
        
    elif optimizer == 'Adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate = learningrate)
        
    elif optimizer == 'Ftrl':
        opt = tf.keras.optimizers.Ftrl(learning_rate = learningrate)
        
    elif optimizer == 'Nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate = learningrate)
        
    elif optimizer == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate = learningrate)
        
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate = learningrate)
        
    #tfa.optimizers

    elif optimizer == 'AdamW':
        opt = tfa.optimizers.AdamW(learning_rate = learningrate)
        
    elif optimizer == 'RectifiedAdam':
        opt = tfa.optimizers.RectifiedAdam(lr = learningrate)

    elif optimizer == 'LAMB':
        opt = tfa.optimizers.LAMB(learning_rate = learningrate)
        
    elif optimizer == 'LazyAdam':
        opt = tfa.optimizers.LazyAdam(learning_rate = learningrate)
        
    elif optimizer == 'SGDW':
        opt = tfa.optimizers.SGDW(learning_rate = learningrate)
          
    elif optimizer == 'Ranger':
        opt = tfa.optimizers.RectifiedAdam(learning_rate = learningrate)
        opt = tfa.optimizers.Lookahead(opt)
        
    elif optimizer == 'LookaheadSGD':
        opt = tf.keras.optimizers.SGD(learning_rate = learningrate)
        opt = tfa.optimizers.Lookahead(opt,)
    

    
    return opt




def Loss(loss_type):
    if loss_type == 'BinaryCrossentropy':
        loss = tf.keras.losses.BinaryCrossentropy()
        
    elif loss_type == 'CategoricalCrossentropy':
        loss = tf.keras.losses.CategoricalCrossentropy()
        
    elif loss_type == 'CategoricalHinge':
        loss = tf.keras.losses.CategoricalHinge()
        
    elif loss_type == 'CosineSimilarity':
        loss = tf.keras.losses.CosineSimilarity()   
        
    elif loss_type == 'Hinge':
        loss = tf.keras.losses.Hinge()
        
    elif loss_type == 'Huber':
        loss = tf.keras.losses.Huber()
        
    elif loss_type == 'KLDivergence':
        loss = tf.keras.losses.KLDivergence()
        
    elif loss_type == 'LogCosh':
        loss = tf.keras.losses.LogCosh()
        
    elif loss_type == 'MeanAbsoluteError':
        loss = tf.keras.losses.MeanAbsoluteError()
        
    elif loss_type == 'MeanSquaredLogarithmicError':
        loss = tf.keras.losses.MeanSquaredLogarithmicError()
        
    elif loss_type == 'Poisson':
        loss = tf.keras.losses.Poisson()
        
    elif loss_type == 'SparseCategoricalCrossentropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy() 
        
    elif loss_type == 'SquaredHinge':
        loss = tf.keras.losses.SquaredHinge() 
    
    return loss