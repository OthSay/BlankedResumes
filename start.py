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
from data_augmentation import prepare_data_seg
# ============ Meta parameters ==================
input_size = 224
icons_size = 32
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

y_binary = np.array([int(label =='nothing') for label in y])

x_add, masks = prepare_data_seg(resumes=x, 
                                labels = y, 
                                resume_size = input_size, 
                                icons_size = icons_size, 
                                icon_per_resume = 2)

final_x = np.concatenate((x,x_add), axis=0)
final_y = np.concatenate((y_binary , np.zeros(x_add.shape[0])))

datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split = 0.1,
        )

train_generator = datagen.flow(final_x,
                      to_categorical(final_y),
                      batch_size = batch_size, 
                      shuffle = True, 
                      subset='training')

validation_generator = datagen.flow(final_x,
                      to_categorical(final_y),
                      batch_size = batch_size, 
                      shuffle = True, 
                      subset='validation')

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(final_y), 
                final_y)


#============ Defining first binary classification model ==================

clf_model = build_vgg_model(input_size=input_size,nb_classes=2)

#clf_model = build_resnet_model(input_size=input_size,nb_classes=2)

adam = Adam(lr = 0.0001)

clf_model.compile(optimizer = adam,
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])

clf_model.fit_generator(train_generator,
                        class_weight=class_weights,
                        shuffle = True,
                        epochs = 10, 
                        validation_data=validation_generator)



