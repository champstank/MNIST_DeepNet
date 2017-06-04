# MNIST_DeepNet
DeepNet implementing Keras with Theano backend applying GridSearch on MNIST dataset with plotting visualization of learning on accuracy loss and accuracy validation.


## Overview
For this exercise, we were given a default script that produced a score of 0.983.  The task was to improve the model workflow and practice tuning parameters to find a better model with higher scoring. 

1. Summary of default script and performance evaluation.
2. Smoothing the workflow with build model function.
3. Tuning parameters with GridSearchCV.
4. Initializing best parameters in model creation.
5. Vizualizing model performance.

### Summary of default script and performance evaluation
Starting with a baseline script that produced a score of 0.983.

First we import all the modules we will need.
```python
'''Trains a simple convnet on the MNIST dataset for ONLY digits 3 and 8.
Gets to 98.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
4 seconds per epoch on a 2 GHz Intel Core i5.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
%matplotlib inline
```
Next we initialize some parameters and the MNIST data is loaded and split into train and test samples.
```python
batch_size = 128
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
Here we find only the 3's and 8's.  Then convert the data into a workable type for the model.  This makes it more reasonable to train on a cpu.
```python
#Only look at 2's and 7's
train_picks = np.logical_or(y_train==2,y_train==7)
test_picks = np.logical_or(y_test==2,y_test==7)

x_train = x_train[train_picks]
x_test = x_test[test_picks]
y_train = np.array(y_train[train_picks]==7,dtype=int)
y_test = np.array(y_test[test_picks]==7,dtype=int)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
```
Next we convert the model into binary classes for classification purposes.
```python
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```
Then we set up our model build with type and layers, then compile the model.
```python
model = Sequential()
model.add(Conv2D(4, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```
Next we fit the model on the data and load it into a **history** variable for easy plotting of performace to visualize model learning.
```python
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
```
Then we print the **accuracy loss** and **accuracy** of the model.
```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
Test loss: 0.0500074749647    
Test accuracy: 0.985922330097
---














