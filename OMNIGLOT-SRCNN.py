#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:34:35 2017

@author: hongbin
"""

from keras.datasets import mnist
from keras.layers import Reshape,Dense
import numpy as np
from keras.layers import UpSampling2D,MaxPool2D,Conv2D,Activation,Dropout,BatchNormalization
from keras.models import Sequential,load_model
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy import ndimage as nd
import myplot as myplot
from skimage.measure import compare_psnr,compare_ssim
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

f= np.load('OMNIGLOT.npz')
HR = f['omni'].astype('uint8')
LR_SRF2 = f['omni_by2'].astype('uint8')
LR_SRF4 = f['omni_by4'].astype('uint8')
LR_SRF8 = f['omni_by8'].astype('uint8')

LR_mnist = LR_SRF2

bicubic_LR = np.reshape(LR_mnist,(LR_mnist.shape[0],LR_mnist.shape[1],LR_mnist.shape[2]))
temp_bicubic= []

for index in tqdm(range(len(bicubic_LR))):
    temp_bicubic.append(imresize(bicubic_LR[index],size=2.0,interp='bicubic'))
bicubic_LR =np.array(temp_bicubic)

myplot.plot_comparison(HR[1],LR_mnist[1])

bicubic_LR = np.reshape(bicubic_LR,(bicubic_LR.shape[0],bicubic_LR.shape[1],bicubic_LR.shape[2],1))

HR = np.reshape(HR,(HR.shape[0],HR.shape[1],HR.shape[2],1))


bicubic_LR = (bicubic_LR-127.5)/128 
                    
HR = (HR-127.5)/128 
losses = []
lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for learing_rate in lrs:
    G=Sequential()
    G.add(Conv2D(64,(9,9),padding='same',input_shape=(None,None,1)))
    #G.add(Conv2D(16,(5,5),padding='same'))
    #G.add(Dropout(0.9))
    G.add(Activation('relu')) 
    G.add(Conv2D(32,(1,1),padding='same'))
    G.add(Activation('relu'))
    # G.add(BatchNormalization())
    G.add(Conv2D(1,(2,2),padding='same'))
    #G.add(Activation('sigmoid'))
    G.summary()
    optim =Adam(lr=learing_rate)
    G.compile(loss='mse',optimizer=optim)
    history = G.fit(x=bicubic_LR,y=HR,batch_size=50,epochs=20)
    the_loss = history.history['loss']
    losses.append(the_loss)



G.save('OMNI_srcnn_SRF2.h5')
G= load_model('OMNI_srcnn_SRF2.h5')
predimage = G.predict(x=bicubic_LR[0:200])

predimage = np.reshape(predimage,(predimage.shape[0],predimage.shape[1],predimage.shape[2]))

HR = np.reshape(HR,(HR.shape[0],HR.shape[1],HR.shape[2]))

predimage[predimage<0.5]=0

predimage[predimage>0.5]=1
# HR=HR.astype('uint8')
predimage =predimage.astype('uint8')
index =66
myplot.plot_3( HR[index], LR_SRF2[index],  predimage[index])
plt.savefig('OMNIGLOT_SRF2_'+str(index))

plt.hist(HR.flatten())
plt.show()


diff_image = HR[index] - predimage[index]
plt.imshow(diff_image,cmap='gray')
plt.show()

plt.hist(diff_image.flatten())
plt.show()
for index in range(3,8):
    plt.plot(loss_array[index])for index in range(3,8):
    plt.plot(loss_array[index])
    plt.plot(loss_array[index])
    plt.legend('lr=0.1') 