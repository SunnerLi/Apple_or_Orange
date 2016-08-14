from test import *
import os
import Image

if __name__ == "__main__":
    model = loadModel("./model/model.h5")
    path = os.listdir("./Img/Demo/")
    for name in path:
        img = Image.open("./Img/Demo/" + name)
        showResult(test(model, img), name)