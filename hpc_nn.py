# -*- coding: utf-8 -*-
"""
Created on Mon April  13 18:25:04 2020

dataset used: https://www.cs.toronto.edu/~kriz/cifar.html
@author: Morgan, Aj

"""
#epochs 1, verbose 1, batchsize 50 had accuracy of 53.86%
#epochs 2, verbose 1, batchsize 50 had accuracy of 70.87%
#epochs 25, verbose 0, batchsize 54 had accuracy of 83.35%
#using the keras cifar10 dataset of pictures, we find the accuracy at which the nn is recognizing the objects as what they are
#we then plot that data and use our own image to test if it is accurate


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import matplotlib.pyplot as plt 

# Set random seed for purposes of reproducibility
seed = 21

from keras.datasets import cifar10

#Loading in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


#Parse numbers as floats & normalize data
X_train = X_train.astype('float32') /255.0
X_test = X_test.astype('float32') / 255.0

#Normalize data
#X_train = X_train / 255.0
#X_test = X_test / 255.0

#one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_of_classes = y_test.shape[1]


#Defining the model
model = Sequential()

#Filter into a 32 filter 3x3 matrix
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))

#Drop any image that is overlapping
model.add(Dropout(0.2))

#Transform inputs, can be added to models to standardize raw input variables or the outputs of a hidden layer
model.add(BatchNormalization())

#Increases to 64, 128, 256 size filter to input more complex images
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

#Pooling allows the discard of a certain amount of information
#if it is too large it will remove too much information
#change pooling size by 2 increments based on image size
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

#kernel constraint can regularize the data as it learns, another thing that helps prevent overfitting. This is why we use maxnorm
model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(num_of_classes))

#softmax activation function selects the neuron with the highest probability as its output, deciding that the image belongs to that class
model.add(Activation('softmax'))

# epochs 25
epochs = 30
optimizer = 'adam'

#Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#print(model.summary())

np.random.seed(seed)

#batch size 54
#Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=100)


#Final Model evaluation
scores = model.evaluate(X_test, y_test, verbose=1) #verbose 0,2
print("Accuracy: %.2f%%" % (scores[1]*100))

#Plot model history
plt.plot(history.history['accuracy'])
plt.show()





