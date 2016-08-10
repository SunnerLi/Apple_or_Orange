import random
import numpy as np
import cPickle
import Image

from keras.models import Sequential
from keras import backend as K

def test(model, img):
    """
        Test the image for the trained model

        Arg   : The keras model and the Image object(Image module)
        Return: Predict numpy array
    """    
    # Load the image
    
    img = img.resize((200, 200))
    img = np.asarray(img, dtype="float32")
    img = img.reshape((3, 200, 200))
    data = np.empty((1, 3, 200, 200), dtype="float32")
    data[0, :, :, :] = img
    data = data.astype('float32')

    # Predict
    predict = K.function([model.layers[0].input, K.learning_phase()], [model.layers[10].output])
    return predict([data, 1])

def loadModel(path):
    """
        Load the model from the specific path

        Return: the keras model
    """
    print "==> Start to load model <=="
    model = cPickle.load(open(path, "rb"))
    print "==> Loading model Done<=="
    return model

def showResult(arr, imgName=None):
    """
        Show the result by text according to the value of the array

        Arg    : The result array
    """
    if arr[0][0][0] > arr[0][0][1]:
        if imgName == None:
            print "It's apple !"
        else:
            print "It's apple in ", imgName
    else:
        if imgName == None:
            print "It's orange !"
        else:
            print "It's orange in ", imgName

def test_demo():
    """
        Testing demo function
    """
    model = loadModel("./model/model.pkl")
    img = Image.open("img1.jpg")
    showResult(test(model, img))

test_demo()