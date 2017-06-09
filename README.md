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
'''Trains a simple convnet on the MNIST dataset for ONLY digits 2 and 7.
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
Here we find only the 2's and 7's.  Then convert the data into a workable type for the model.  This makes it more reasonable to train on a cpu.
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

We can preview our images to see what the net will see.  To do this we need to use numpy's squeeze() function to reduce the dimensionality of the image for plot to be able to handle.
```python
plt.imshow(np.squeeze(x_train[0]), cmap='gray')
```

<img width="248" alt="two" src="https://user-images.githubusercontent.com/8240939/26986431-c0fa5bf4-4d04-11e7-8eb5-778163e103f7.png"> | <img width="251" alt="seven" src="https://user-images.githubusercontent.com/8240939/26986267-fd769d28-4d03-11e7-9ab8-06cc1e2e8783.png">


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
## Smoothing the workflow with build model function
First thing to improve our script we build a function to initialize our Keras model.  This makes it easier to pass parameters from our future parameter tuning techniques.
```python
#build Keras model
def create_model():
    #create model
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    #compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    
    return model
```
We can call this function with the following bit of code.
```python
model = KerasClassifier(build_fn=create_model)
```
##  Tuning parameters with GridSearchCV
Next we started using **GridSearchCV** to find the optimal parameters for the model.  Since we are working with CPU's we approached this as a coarse pass with less paramters at once.  Meaning the model was not searched as exhaustively since it took to much time on the CPU's.  Everytime we ran it we narrowed the params we had already tuned and added new ones to search.  This approach ended up improving the score consistently above 0.993.

A list of parameters we ended up tuning and a range of each were.
```python
#size of data to grab at a time
batch_size = 128
#loops on dataset
epochs = 12
#optimizers
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#learning rate
learn_rate = [0.001, 0.01, 0.1]
#activation for first layer
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#initializer 
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#weight
weight_constraint = [1, 3, 5]
#dropout percent
dropout_rate = [0.25,0.5.9]
# # of neurons
neurons = [25,30,35]
```
Again, we narrowed these down, searching only two or three at a time.  The goal was to stay under a 1000 fits maximum at a time.  So to execute this approach we only provide the parameters we want to pass through the **param_grid** option of **GridSearchCV** as following.
```python
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, and so on.......)
```
What ever variables we pass through **GridSearchCV** we also need to pass them into the **create_model** function as such.
```python
def create_model(learn_rate=learn_rate, optimizer=optimizer, and so on.......):
```
Once we have found the value we want to use for the variable we can hardcode it into the model creation and don't have to pass it in.


Next, to initialize and instantiate **GridSearchCV()** with the **Keras** model we do the following.
```python
grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=-1,verbose=1)
```
**n_jobs=-1** runs the job on all cores to speed it up and **verbose=1** prints out how many fits we will be doing.

Then to run exhaustive **GridSearchCV** on parameters evaluate performance of different tunings we do
```python
grid = grid_search.fit(x_train,y_train)
```
Now that we have performed these steps we can pull out the best score and our best model with optimal parameter values.
```python
# #best score
best_score = grid.best_score_
# #best params
best_params = grid.best_params_

# #print out results
print ('Grid Score = ' , best_score)
print ('Grid Best Parameters = ' , best_params)
```
Grid Score =  0.994518530351    
Grid Best Parameters =  {'dropout_rate': 0.25, 'learn_rate': 0.002, 'weight_constraint': 3, 'batch_size': 128, 'epochs': 12, 'neurons': 30, 'init_mode': 'lecun_uniform'}


## Initializing best parameters in model creation
Now that we know our best parameters we can code them directly into the **create_model** function.  Notice again we are not passing in any parameters into the model since we are no longer tuning parameters.
```python
def create_model():
    #create model
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(25, kernel_initializer='lecun_uniform', activation='linear', kernel_constraint=maxnorm(3))) 
    model.add(Dropout(0.5))
    model.add(Dense(2, kernel_initializer='lecun_uniform', activation='softmax'))
    
    #initialize optimizer
    nadam = Nadam()
    
    #compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=nadam,
              metrics=['accuracy'])
    
    return model
```

## Vizualizing model performance
An easy way to evaluate model performance without trying to hard is to load the output of **model.fit()** into a variable as such
```python
history = model.fit((x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
```
This allow a vizualization plot on the **accuracy loss** and **accuracy** fairly easy with the following code.
```python
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
This gives the output

