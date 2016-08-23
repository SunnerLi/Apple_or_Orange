import matplotlib.pyplot as plt
import os

x = [i for i in range(0, 30)]
loss = [0.7227, 0.6883, 0.6902, 0.6977, 0.6953, 0.6920, 0.6925
    , 0.6918, 0.6912, 0.6929, 0.6925, 0.6783, 0.6605, 0.5897, 0.4070
    , 0.2655, 0.3993, 0.2792, 0.1922, 0.2087, 0.1869, 0.1702, 0.1882
    , 0.2153, 0.2356, 0.1609, 0.1698, 0.1463, 0.1229, 0.1137]
acc = [0.5194, 0.5611, 0.6056, 0.5000, 0.5111, 0.5417, 0.5361
    , 0.5333, 0.5111, 0.5278, 0.5639, 0.6194, 0.6417, 0.7111, 0.8056
    , 0.8861, 0.8639, 0.8944, 0.9361, 0.9222, 0.9361, 0.9361, 0.9471
    , 0.9000, 0.9222, 0.9389, 0.9472, 0.9417, 0.9528, 0.9611]
val_loss = []
val_acc = []

def getData(path, round):
    """
        Get the result data from txt file
    """
    fd = open(path, 'r')
    loss = []
    acc = []
    for i in range(round):
        values = []
        _ = fd.readline()
        _ = fd.readline()
        values = _.split('-')
        loss.append(float(values[2][7:13]))
        acc.append(float(values[3][6:12]))
        val_loss.append(float(values[4][11:17]))
        val_acc.append(float(values[5][10:16]))
        #print loss[i], '\t', acc[i], '\t', val_loss[i], '\t', val_acc[i]

def plot():
    """
        Draw the result.
    """
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    
    plt.subplot(221)
    p = plt.plot(x, loss, 'o')
    plt.xlabel("epochs")
    plt.ylabel("percentage")
    plt.title("train loss")

    plt.subplot(222)
    p = plt.plot(x, acc, 'o')
    plt.xlabel("epochs")
    plt.ylabel("percentage")
    plt.title("train accuracy")

    plt.subplot(223)
    p = plt.plot(x, val_loss, 'o')
    plt.xlabel("epochs")
    plt.ylabel("percentage")
    plt.title("validation loss")

    plt.subplot(224)
    p = plt.plot(x, val_acc, 'o')
    plt.xlabel("epochs")
    plt.ylabel("percentage")
    plt.title("validation accuracy")
    plt.show()

if __name__ == "__main__":
    getData("./number1.txt", 30)
    plot()
