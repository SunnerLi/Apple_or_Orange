from loader import load_img
from keras.layers.core import Dropout, Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
import numpy as np

def create_sunnerNet():
    """
        Use keras to create CNN

        Return: the model object of keras
    """
    # 1st Convolution layer
    model = Sequential()
    model.add(Convolution2D(48, 11, 11, border_mode='valid', input_shape=(3, 200, 200), subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # 2nd Convolution layer
    model.add(Convolution2D(128, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    # 3rd Convolution layer
    model.add(Convolution2D(192, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4rd Convolution layer
    model.add(Convolution2D(80, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Dense
    model.add(Flatten())
    model.add(Dense(1000, init='normal'))
    model.add(Activation('tanh'))

    # Softmax
    model.add(Dense(2, init='normal'))
    model.add(Activation('softmax'))

    print model.summary()
    return model

def create_Perceptron():
    """
        Use keras to create perceptron 

        Return: the model object of keras
    """
    # input layer
    model = Sequential()
    model.add(Flatten(input_shape=(3, 200, 200)))
    model.add(Dense(500, init='normal'))
    model.add(Activation('sigmoid'))

    # hidden layer
    model.add(Dense(100, init='normal'))
    model.add(Activation('sigmoid'))

    # Softmax
    model.add(Dense(2, init='normal'))
    model.add(Activation('softmax'))

    print model.summary()
    return model

def create_tinyPerceptron():
    """
        Use keras to create tiny-perceptron 

        Return: the model object of keras
    """
    # input layer
    model = Sequential()
    model.add(Flatten(input_shape=(3, 200, 200)))
    model.add(Dense(192, init='normal'))
    model.add(Activation('sigmoid'))

    # 1st hidden layer
    model.add(Dense(96, init='normal'))
    model.add(Activation('sigmoid'))

    # Softmax
    model.add(Dense(2, init='normal'))
    model.add(Activation('softmax'))

    print model.summary()
    return model

def create_fullCIFAR10():
    """
        Use keras to create CIFAR-10 archetecture 

        Return: the model object of keras
    """
    # 1st Convolution layer
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(3, 200, 200)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # 2nd Convolution layer
    model.add(Convolution2D(32, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())

    # 3rd Convolution layer
    model.add(Convolution2D(64, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    # Dense
    model.add(Flatten())
    model.add(Dense(250, init='normal'))
    model.add(Activation('tanh'))

    # Softmax
    model.add(Dense(2, init='normal'))
    model.add(Activation('softmax'))

    print model.summary()
    return model

def init_tensorflowCIFAR(shape, name=None):
    """
        Callback function defined in github to initialize as 0.1
    """
    m = np.ones(shape)
    return K.variable(m/10, name=name)

def create_tensorflowCIFAR10():
    """
        Use keras to create CIFAR-10 built in tensorflow github example

        Return: the model object of keras
    """
    # 1st Convolution layer
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(3, 200, 200), bias=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())

    # 2nd Convolution layer
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    # 1st Dense
    model.add(Flatten())
    model.add(Dense(384, init=init_tensorflowCIFAR))
    model.add(Activation('relu'))

    # 2nd Dense
    model.add(Dense(192, init=init_tensorflowCIFAR))
    model.add(Activation('relu'))

    # Softmax
    model.add(Dense(2, init='zero'))
    model.add(Activation('softmax'))

    print model.summary()
    return model

def create_smallTensorflowCIFAR10():
    """
        Use keras to create CIFAR-10 built in tensorflow github example

        Return: the model object of keras
    """
    # 1st Convolution layer
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(3, 200, 200), bias=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())

    # 2nd Convolution layer
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    # 1st Dense
    model.add(Flatten())
    model.add(Dense(384, init=init_tensorflowCIFAR))
    model.add(Activation('relu'))

    # 2nd Dense
    model.add(Dense(192, init='normal'))
    model.add(Activation('relu'))

    # Softmax
    model.add(Dense(2, init='normal'))
    model.add(Activation('softmax'))

    print model.summary()
    return model

def create_VGG16(weights_path=None):
    """
        Use keras to create VGG16 example

        Return: the model object of keras
    """
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(3, 200, 200)))
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    print model.summary()
    return model