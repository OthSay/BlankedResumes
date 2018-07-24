#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 23:35:56 2018

@author: sayemothmane
"""
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import Model


def build_fcn_vgg_32(input_shape = (224, 224, 3), nb_classes = 2):
    
    vgg_model = VGG16(
            include_top = False,
            weights='imagenet',
            input_shape = input_shape
            )
        
    vgg_output = Conv2D(4096,
                        (7,7), 
                       activation = 'relu', 
                       padding = 'same')(vgg_model.output)
    vgg_output = Dropout(0.5)(vgg_output)
 
    vgg_output = Conv2D(4096,
                        (7,7),
                       activation = 'relu', 
                       padding = 'same')(vgg_output)
    vgg_output = Dropout(0.5)(vgg_output)

    vgg_output = Conv2D(nb_classes,
                        (1,1),
                        activation = 'linear',
                        padding='valid', 
                        strides=(1, 1))(vgg_output)

    vgg_output = Conv2DTranspose(nb_classes, 
                                 kernel_size=(64,64) ,  
                                 strides=(32,32)
                                )(vgg_output)
    
    vgg_output = Reshape(input_shape)(vgg_output)
	vgg_output = Permute((2,1))
                            (vgg_output)
	vgg_output = Activation('softmax')(vgg_output)
    
	final_model = Model( vgg_model.input , vgg_output )
    
    
    return final_model