#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 16:14:18 2018

@author: sayemothmane
"""
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from toolbox import prepare_data, build_vgg_model
from keras.optimizers import Adam
from sklearn.utils import class_weight

# ============ Meta parameters ==================
input_size = 128
batch_size = 32


x , y = prepare_data(input_size)

print('Number of resumes with company : ' , sum(y=='has_company'))
print('Number of resumes with icon : ' , sum(y=='has_icon'))
print('Number of resumes with both : ' , sum(y=='has_both'))
print('Number of resumes with nothing : ' , sum(y=='nothing'))

'''
     We'll transform the task to a binary classification
     1  : has no company, no icon logo, nothing 
     0 : has either a company logo, an icon, or both 
     still a very unbalanced problem... 
'''

x_add, masks = prepare_data_seg(resume_size = input_size, 
                                icons_size = 12, 
                                icon_per_resume = 1)


y_binary = np.array([int(label =='nothing') for label in y])


datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split = 0.1,
        )

train_generator = datagen.flow(x,
                      to_categorical(y_binary),
                      batch_size = batch_size, 
                      shuffle = True, 
                      subset='training')

validation_generator = datagen.flow(x,
                      to_categorical(y_binary),
                      batch_size = batch_size, 
                      shuffle = True, 
                      subset='validation')

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(y_binary), 
                y_binary)


#============ Defining first VGG16 model ==================

clf_model = build_vgg_model(input_size=input_size,nb_classes=2)

adam = Adam(lr = 0.0001)

clf_model.compile(optimizer = adam,
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])

clf_model.fit_generator(train_generator,
                        class_weight=class_weights,
                        shuffle = True,
                        epochs = 10, 
                        validation_data=validation_generator)
