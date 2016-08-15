import random
import numpy as np
import cPickle
import sys
from model import *

from loader import load_img
from keras.layers.core import Dropout, Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential

trainNumber = 100
sys.setrecursionlimit(10000)

"""
    Main function
"""
# Prepare data and labels
print "==> Start to load training data"
data, label = load_img()
label = np_utils.to_categorical(label, 2)
print "==> Finish loading data"

# Build the model
model = create_tinyPerceptron()
opt_method = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt_method, metrics=['accuracy'])
print "==> Finish create model"

# Shuffle the training data
index = [ i for i in range(len(data)) ]
random.shuffle(index)
data = data[index]
label = label[index]
(x_train, x_val) = (data[0:trainNumber], data[trainNumber:])
(y_train, y_val) = (label[0:trainNumber], label[trainNumber:])

# Train
model.fit(x_train, y_train, batch_size=20, 
    validation_data=(x_val, y_val), nb_epoch=150)
print "==> Finish training"
model.save('./model.h5')
