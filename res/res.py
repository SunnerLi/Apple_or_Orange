import matplotlib.pyplot as plt
import os

x = [i for i in range(0, 50)]
loss = []
acc = []
val_loss = []
val_acc = []

def getData(path, round):
    """
        Get the result data from txt file
    """
    global loss, acc, val_acc, val_loss

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
    getData("./number1.txt", 50)
    plot()
