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
