#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:08:38 2018

@author: sayemothmane
"""


import numpy as np 
import glob
from PIL import Image
import json 
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Dense, Flatten
from keras.models import Model

def prepare_data(input_size = 224):    
    with open('labels.json') as json_data:
        labels = json.load(json_data)
    
    image_list = []
    y=[]

    for filename in glob.glob('labelled-resumes/*.png'): #assuming gif
        im=np.array(Image.open(filename).resize((input_size, input_size)))
        y.append(labels[filename[filename.find('/')+1:]])
        image_list.append(im)
        
    image_list = np.array(image_list)
    y = np.array(y)
    
    return image_list, y 


def build_vgg_model(input_size = 224, nb_classes = 2):
    
    vgg_model = VGG16(
            include_top = False,
            weights='imagenet',
            input_shape = (input_size, input_size, 3)
            )
    
    vgg_output = Flatten()(vgg_model.output)
    
    vgg_output = Dense(4096,
                       activation = 'relu')(vgg_output)
    vgg_output = Dropout(0.5)(vgg_output)
 
    vgg_output = Dense(4096,
                       activation = 'relu')(vgg_output)
    vgg_output = Dropout(0.5)(vgg_output)

    vgg_output = Dense(nb_classes,
                       activation = 'relu')(vgg_output)

    final_model = Model(vgg_model.input, vgg_output)
    
    return final_model


