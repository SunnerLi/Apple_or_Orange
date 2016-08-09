import random
import numpy as np
import cPickle

from loader import load_img
from keras.layers.core import Dropout, Activation, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential

def createCNN():
    """
        Use keras to create CNN

        Return: the model object of keras
    """
    # 1st Convolution layer
    model = Sequential()
    model.add(Convolution2D(8, 5, 5, border_mode='valid', input_shape=(200, 200, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution layer
    model.add(Convolution2D(16, 10, 10, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd Convolution layer
    model.add(Convolution2D(32, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4rd Convolution layer
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense
    model.add(Flatten())
    print model.summary()
    model.add(Dense(32, init='normal'))
    model.add(Activation('tanh'))

    # Softmax
    model.add(Dense(2, init='normal'))
    model.add(Activation('softmax'))
    return model

"""
    Main function
"""
# Prepare data and labels
print "==> Start to load training data"
data, label = load_img()
label = np_utils.to_categorical(label, 2)
print "==> Finish loading data"

# Build the model
model = createCNN()
opt_method = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt_method, metrics=['accuracy'])
print "==> Finish create model"

# Shuffle the training data
index = [ i for i in range(len(data)) ]
random.shuffle(index)
data = data[index]
label = label[index]
(x_train, x_val) = (data[0:55], data[55:])
(y_train, y_val) = (label[0:55], label[55:])

# Train
model.fit(x_train, y_train, batch_size=5, 
    validation_data=(x_val, y_val), nb_epoch=10)
print "==> Finish training"
cPickle.dump(model, open("./model.pkl", "wb"))
