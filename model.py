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

def init_tensorflowCIFAR_beta(shape, name=None):
    """
        Callback function defined in github to initialize as 0.75
    """
    m = np.ones(shape)
    return K.variable(3*m/4, name=name)

def init_tensorflowCIFAR_gamma(shape, name=None):
    """
        Callback function defined in github to initialize as 0.001/9.0
    """
    m = np.ones(shape)
    return K.variable(m/9000, name=name)

def create_tensorflow_CIFAR10():
    """
        Use keras to create CIFAR-10 built in tensorflow github example

        Return: the model object of keras
    """
    # 1st Convolution layer
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(3, 200, 200), bias=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization(beta_init=init_tensorflowCIFAR_beta, gamma_init=init_tensorflowCIFAR_gamma))

    # 2nd Convolution layer
    model.add(Convolution2D(64, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(beta_init=init_tensorflowCIFAR_beta, gamma_init=init_tensorflowCIFAR_gamma))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    # 1st Dense
    model.add(Flatten())
    model.add(Dense(384, init=init_tensorflowCIFAR, bias=True))
    model.add(Activation('relu'))

    # 2nd Dense
    model.add(Dense(192, init=init_tensorflowCIFAR, bias=True))
    model.add(Activation('relu'))

    # Softmax
    model.add(Dense(2, init='zero'))
    model.add(Activation('softmax'))

    print model.summary()
    return model

def create_Tensorflow_smallCIFAR10():
    """
        Use keras to create CIFAR-10 built in tensorflow github example

        Return: the model object of keras
    """
    # 1st Convolution layer
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(3, 200, 200), bias=True))
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

def create_Keras_CIFAR10():
    """
        Use keras to create CIFAR10 example

        Return: the model object of keras
    """
    model = Sequential()

    # 1st Convolution layer
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 200, 200)))
    model.add(Activation('relu'))

    # 2nd Convolution layer
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    # 4rd Convolution layer
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Dense
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Softmax
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    print model.summary()
    return model


def create_Keras_VGG16(weights_path=None):
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

def create_LeNet(weights_path=None):
    """
        Use keras to create LeNet structure

        Return: the model object of keras
    """
    model = Sequential()

    # 1st Convolution layer
    model.add(Convolution2D(8, 28, 28, border_mode='same', input_shape=(3, 200, 200)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # 2nd Convolution layer
    model.add(Convolution2D(20, 10, 10, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # Dense
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    print model.summary()
    return model