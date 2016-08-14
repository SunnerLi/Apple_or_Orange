import os
import Image
import numpy as np

def load_img():
    """
        Load training image

        Return: The labels and the images
    """
    # Create the temporary folder
    if not os.path.exists("./temp"):
        os.mkdir("./temp")

    # Resize the images to the temporary folder
    imgs = os.listdir("./Img/Train/All/")
    for i in range(len(imgs)):
        img = Image.open("./Img/Train/All/" + imgs[i])
        img = img.resize((200, 200))
        img.save("./temp/" + imgs[i])

    # Load the images again and product the labels
    datas = np.empty((147, 3, 200, 200), dtype="float32")
    labls = np.empty((147), dtype="uint8")
    imgs = os.listdir("./temp")
    print len(imgs)
    for i in range(len(imgs)):
        img = Image.open("./temp/" + imgs[i])
        img = np.asarray(img, dtype="float32")
        img = img.reshape((3, 200, 200))
        datas[i, :, :, :] = img
        labls[i] = int( imgs[i].split('.')[0] )
    print np.shape(datas)
    return datas, labls

load_img()