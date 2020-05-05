# -*- coding: utf-8 -*-
"""
Created on Mon April  13 18:25:04 2020

dataset used: https://www.cs.toronto.edu/~kriz/cifar.html
@author: Morgan

"""
#epochs 25, verbose 0, batchsize 54 had accuracy of 83.35%
#using the keras cifar10 dataset of pictures, we find the accuracy at which the nn is recognizing the objects as what they are
#we then plot that data and use our own image to test if it is accurate


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.models import load_model 
from keras.preprocessing import image
import cv2
import matplotlib.image as mping 
import matplotlib.pyplot as plt 

# Set random seed for purposes of reproducibility
seed = 21

from keras.datasets import cifar10

#Loading in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


#Parse numbers as floats 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

#one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_of_classes = y_test.shape[1]

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(num_of_classes))
model.add(Activation('softmax'))

# epochs 25
epochs = 100
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

np.random.seed(seed)
#batch size 54
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=50)

#Final Model evaluation
scores = model.evaluate(X_test, y_test, verbose=1) #verbose 0
print("Accuracy: %.2f%%" % (scores[1]*100))

#Saving Model
model.save('hpc_nn_model.h5')

#Loading the saved model 
model = load_model('hpc_nn_model.h5')

#Testing the model with random image inputs
test_img1 = mping('elk-lying-on-green-grasses-34485.jpg',target_size =(32,32))

#Running random image against all Cifar-10 object classes
test_image =image.img_to_array(test_img1) 
test_image =np.expand_dims(test_image, axis =0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Aeroplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')
    
plt.imshow(test_img1)



