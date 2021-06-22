#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

def import_model(model, pretrain,pool):
    net=[]
    if model == 'DesNet121':
        net = tf.keras.applications.DesNet121(include_top=False,pooling = pool,
                                              weights = pretrain)
    elif model == 'DesNet169':
        net = tf.keras.applications.DesNet169(include_top=False,pooling = pool,
                                              weights = pretrain)
    elif model == 'DesNet201':
        net = tf.keras.applications.DesNet201(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'EfficientNetB0':
        net = tf.keras.applications.EfficientNetB0(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'EfficientNetB1':
        net = tf.keras.applications.EfficientNetB1(include_top=False,pooling = pool,
                                     weights = pretrain) 
    elif model == 'EfficientNetB2':
        net = tf.keras.applications.EfficientNetB2(include_top=False,pooling = pool,
                                     weights = pretrain) 
    elif model == 'EfficientNetB3':
        net = tf.keras.applications.EfficientNetB3(include_top=False,pooling = pool,
                                     weights = pretrain) 
    elif model == 'EfficientNetB4':
        net = tf.keras.applications.EfficientNetB4(include_top=False,pooling = pool,
                                     weights = pretrain) 
    elif model == 'EfficientNetB5':
        net = tf.keras.applications.EfficientNetB5(include_top=False,pooling = pool,
                                     weights = pretrain) 
    elif model == 'EfficientNetB6':
        net = tf.keras.applications.EfficientNetB6(include_top=False,pooling = pool,
                                     weights = pretrain) 
    elif model == 'EfficientNetB7':
        net = tf.keras.applications.EfficientNetB7(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'InceptionResNetV2':
        net = tf.keras.applications.InceptionResNetV2(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'InceptionV3':
        net = tf.keras.applications.InceptionV3(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'MobileNet':
        net = tf.keras.applications.MobileNet(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'MobileNetV2':
        net = tf.keras.applications.MobileNetV2(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'MobileNetV3Large':
        net = tf.keras.applications.MobileNetV3Large(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'MobileNetV3Small':
        net = tf.keras.applications.MobileNetV3Small(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'NASNetLarge':
        net = tf.keras.applications.NASNetLarge(include_top=False,pooling = pool,
                                     weights = pretrain)
        
    elif model == 'NASNetMobile':
        net = tf.keras.applications.NASNetMobile(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'ResNet101':
        net = tf.keras.applications.ResNet101(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'ResNet101V2':
        net = tf.keras.applications.ResNet101V2(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'ResNet152':
        net = tf.keras.applications.ResNet152(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'ResNet152V2':
        net = tf.keras.applications.ResNet152V2(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'ResNet50':
        net = tf.keras.applications.ResNet50(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'ResNet50V2':
        net = tf.keras.applications.ResNet50V2(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'VGG16':
        net = tf.keras.applications.VGG16(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'VGG19':
        net = tf.keras.applications.VGG19(include_top=False,pooling = pool,
                                     weights = pretrain)
    elif model == 'Xception':
        net = tf.keras.applications.Xception(include_top=False,pooling = pool,
                                     weights = pretrain)
    
    return net