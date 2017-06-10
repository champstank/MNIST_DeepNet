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

We can preview our images to see what the net will see.  To do this we need to use numpy's squeeze() function to reduce the dimensionality of the image for plot to be able to handle them.
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

Then we set up our model build with type and layers, these consist of convolutional and activation layers with hidden layers for dropout.  We then compile the model to initialize it as ready.
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

Next we fit the model on the data and load it into a **history** variable to call later for easy plotting of performace to visualize how the model did learning.
```python
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
```

We can print the **accuracy loss** and **accuracy** of the model.
```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
Test loss: 0.0500074749647     
Test accuracy: 0.985922330097 

---
## Smoothing the workflow with build model function
First thing we can do to improve our script is build a function to initialize our Keras model.  This makes it easier to pass parameters for our future parameter tuning techniques.
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
Next we start using **GridSearchCV** to find the optimal parameters for the model.  Since we are working with CPU's we will approach this as a coarse pass with less paramters at once.  Meaning the model was not searched as exhaustively since it took too much time on the CPU's.  Everytime we ran it we narrowed the parameters down we had already tuned and added new ones to search.  This approach ended up improving the score consistently above 0.993.

A list of parameters we ended up tuning and a range of each were.
```python
#size of data to grab at a time
batch_size = 128
#loops on dataset
epochs = 12
#optimizers
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#learning rate
#learn_rate = [0.001, 0.01, 0.1]
#activation for first layer
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#initializer 
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#weight
#weight_constraint = [1, 3, 5]
#dropout percent
#dropout_rate = [0.25,0.5.9]
# # of neurons
neurons = [25,30,35,75,125]
```

Again, we narrowed these down, searching only two or three at a time with the others not used commented out.  The goal was to stay under a 1000 fits maximum at a time.  So to execute this approach we only provide the parameters we want to pass through the **param_grid** option of **GridSearchCV** as following.  If we wanted to tune **batch_size, epochs, optimizer, neurons and activation** we would pass them as such
```python
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, neurons=neurons, activation=activation)
```

Which ever variables we pass through **GridSearchCV** we also need to pass them into the **create_model** function build. We do not need to pass **batch_size and epochs** though.  If we were using the above parameters we would pass and intialize them like this
```python
def create_model(learn_rate=learn_rate, optimizer=optimizer, neurons=neurons, activation=activation):
    #create model
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3),activation=activation,input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    #compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])
    
    return model
```
Notice above we did not pass the **activation** parameter to the last layer.  This layer requires a **sigmoid** activation for the purposes of the problem being binary classification.


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

Another approach if we do not want to wait for **GridSearchCV** to do an exhaustive search is to implement **RandomizedSearchCV**.  This method will randomly search the parameters in the provided dictionary and give us scores close to what GridSearchCV usually provides.

```python
#n_iter must be no higher than the numner of parameters to check
n_iter = 5

#instatiate and test RandomizedSearch for time and score against GridSearchCV
rand_search = RandomizedSearchCV(model,param_distributions=param_grid, cv=3, n_iter=n_iter, verbose=1)

#fit RandomizedSearchCV model
rand = rand_search.fit(x_train,y_train)

#best score
best_score = rand.best_score_
#best params
best_params = rand.best_params_

#print out results
print ('Randomized Score = ' , best_score)
print ('Randomized Best Parameters = ' , best_params)
```
Randomized Score =  0.993127709767
Randomized Best Parameters =  {'epochs': 24, 'learn_rate': 0.001, 'optimizer': 'Adagrad', 'batch_size': 128, 'dropout_rate': 0.25, 'weight_constraint': 3}

After running **GridSearchCV** and **RandomizedSearchCV** we can see the difference in performance is minimal. **GridSearchCV** does give us an edge as we would expect since it performs its search exhaustively, but **RandomizedSearchCV** gives us a score close to the same.


Once we have found the values we want to use for the variables we can hardcode them into the model creation and don't have to pass them in any longer.

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
    model.add(Dense(30, kernel_initializer='lecun_uniform', activation='linear', kernel_constraint=maxnorm(3))) 
    model.add(Dropout(0.5))
    model.add(Dense(2, kernel_initializer='lecun_uniform', activation='softmax'))
    
    #initialize optimizer
    nadam = Nadam(lr=0.002)
    
    #compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=nadam,
              metrics=['accuracy'])
    
    return model
```


## Vizualizing model performance
An easy way to evaluate model performance without trying to hard is to load the output of **model.fit()** into a variable called **history**, we will have to change **epochs** and **batch_size** from a **list** format to an **integer** or the fit will fail. We pass **x_test** and **y_test** into the function through the **validation_data** parameter.
**i.e.** 
epochs = [12]      --> epochs = 12    
batch_size = [128] --> batch_size = 128
```python
history = model.fit((x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
```
The following is output from running the above command

<img width="977" alt="screen shot 2017-06-10 at 10 49 00 am" src="https://user-images.githubusercontent.com/8240939/27004679-8700bf3a-4dca-11e7-8cd7-cdc9b043ab5e.png">


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
Calling plot on **history** gives will show us our model performace in the following output

<img width="424" alt="screen shot 2017-06-10 at 11 32 40 am" src="https://user-images.githubusercontent.com/8240939/27004950-91917736-4dd0-11e7-94af-bf034f196c1f.png">

**Note:**  You can see from the above our model is predicting with **0.9908% accuracy** when it ends and is not converging.  We are running lightly with only **12 epochs**.  If we were digging into the model more intensely we would be testing more epochs, batch_size and learning rate configurations against each other for a better score and would see the model converge.


To see what the two **Convolutional Layers** are seeing we can print out the **weights** of each layer and get a feel for what the model is giving value to in each image. 

To plot and view the weights of the first layer we run the following bit of code
```python
weight = model.model.layers[0].get_weights()[0][:,:,0,:]
plt.figure(1, figsize=(25,15))

for i in range(0,4):
    plt.subplot(1,8,i+1)
    plt.title('Filter #' + str(i+1))
    plt.imshow(weight[:,:,i],interpolation="nearest",cmap="gray")

print('\033[1m' + '1st Convolutional Layer')
plt.show()
```
