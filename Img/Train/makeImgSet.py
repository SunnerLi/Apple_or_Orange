import os
import Image

def main(prefix, imgList):
    """
        Change the image name with given prefix

        Arg    : The given prefix
    """
    imgCount = 0
    for name in imgList:
        if prefix == 0:
            img = Image.open("./apple/" + name)
        else:
            img = Image.open("./orange/" + name)
        print "./All/" + str(prefix) + '.' + str(imgCount)
        img.save("./All/" + str(prefix) + '.' + str(imgCount) + ".jpg")
        imgCount += 1

        

# Make orange images into folder
orange_name_list = os.listdir("./orange")
main(1, orange_name_list)

# Make apple images into folder
apple_name_list = os.listdir("./apple")
main(0, apple_name_list)