#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 18:32:02 2018

@author: sayemothmane
"""

import numpy as np 
import glob
from PIL import Image
from toolbox import prepare_data
import matplotlib.pyplot as plt 
import random

def prepare_data_seg(resume_size = 224, 
                     icons_size = 24, 
                     icon_per_resume = 3):
    
    icons_sn_path  = '/Users/sayemothmane/Documents/MVA/Riminder/icons/145797-social-network-logo-collection/png'
    icons_es_path = '/Users/sayemothmane/Documents/MVA/Riminder/icons/png'
    icons = []
    
    for filename in glob.glob(icons_sn_path+'/*.png'): #assuming gif
            im=Image.open(filename).resize((icons_size, icons_size))
            im = np.array(im.convert("RGB"))
            im[im==0]=255
            icons.append(im)  
    for filename in glob.glob(icons_es_path+'/*.png'): #assuming gif
            im=Image.open(filename).resize((icons_size, icons_size))
            im = np.array(im.convert("RGB"))
            #im[im==0]=255
            icons.append(im)
        
    icons = np.array(icons)


    x , y = prepare_data(resume_size)

    empty_resumes = x[y=='nothing', :, :,:]
        
    margin = 10
    new_resumes = []
    masks=[]
    for resume in empty_resumes: 
        icons_list = random.sample(range(len(icons)), icon_per_resume)
        for icon in icons_list : 
            mask  = np.zeros((resume_size, resume_size))
            pos_w = random.randint(margin, resume_size-icons_size-margin)
            pos_l = random.randint(margin, resume_size-icons_size-margin)
            new_resume = np.array(resume, copy = True)
            new_resume[pos_l:icons_size+pos_l, pos_w:icons_size+pos_w, :] = icons[icon, :, :]
            mask[pos_l:icons_size+pos_l, pos_w:icons_size+pos_w] = 1
            new_resumes.append(new_resume)
            masks.append(mask)
        
    masks = np.array(masks)
    new_resumes = np.array(new_resumes)
    
    return new_resumes, masks 