"""
    Written by SunnerLi at 2016/8/14.
    This script can reformat the trainning image as the specific order.
    What's more, it can eliminate the duplicate images.
"""
import os
import Image
import numpy as np

imgSet = set([])

def rmAll():
    """
        Remove all content in the All folder
    """
    if os.path.exists("./All"):
        for name in os.listdir("./All"):
            os.remove(name)
    else:
        os.mkdir("./All")
        
def listdir(path):
    """
        Return the list of the content in the folder with prefix
        
        Return : The list of the whole name 
    """
    imgList = os.listdir(path)
    imgList_ = []
    for name in imgList:
        imgList_.append( path + "/" + name )
    return imgList_

def main(prefix, imgList):
    """
        Change the image name with given prefix

        Arg    : The given prefix and the images path list
    """
    imgCount = 0
    main_printRest("Origin", len(imgList))
    for name in imgList:
        img = Image.open(name)
        # If the image isn't exist, save it!
        exist_, imgTuple = isExist(img)
        if exist_:  
            main_printSkip("Origin", name, imgTuple, len(imgList)-imgCount)
        else:
            img.save("./All/" + str(prefix) + '.' + str(imgCount) + ".jpg")
        imgCount += 1
    print "Finish Merge image !!"
    rotate(prefix, imgList, imgCount)
        
def isExist(img):
    """
        Judge if the image is probably exist
        
        Arg    : The image instance
        Return : If exist 
    """
    # Calculate the sum of R, G, B brand
    # Additionally, it calculate the gradient of the R brand
    #   to recognize the horizential flip.
    pixelSum = 0
    r = 0
    g = 0
    b = 0
    for band_index, band in enumerate(img.getbands()):
        s = np.array([p[band_index] for p in img.getdata()]).reshape(*img.size)
        if band_index == 0:
            r = s
        elif band_index == 1:
            g = s
        elif band_index == 2:
            b = s
    imgTuple = (np.sum(r), np.sum(g), np.sum(b), np.sum(np.gradient(r)))
    
    # Judge if the image had been in set       
    sizeBefore = len(imgSet)
    imgSet.add(imgTuple)
    sizeAfter = len(imgSet)
    if sizeAfter == sizeBefore:
        return True, imgTuple
    return False, imgTuple
    
def main_printSkip(pName, name, tup, rest):
    """
        Print the skip info with name and image tuple value
        
        Arg    : The image name, tuple value and rest index 
    """
    main_printRest(pName, rest)
    print "{", pName, "} Skip image:\t", (name)
    print "{", pName, "} Tuple value:\t", tup
    print "-----------------------------------------------------------------"
    
def main_printRest(pName, rest):
    """
        Print rest time
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print "{", pName, "} Estimate: ", str(rest*2), "seconds need\n\n"
    print "-----------------------------------------------------------------"
        
def rotate(prefix, imgList, imgCount):
    """
        Rotate the each image to vary the training data
        
        Arg    : The image prefix, the images path and the last index of the image 
    """
    """
        imgCount    : to keep count the image number that help rename the image
        imgNewCount : to help record the number of rest images
    """
    imgNewCount = 0
    
    main_printRest("Rotate", len(imgList))
    for name in imgList:
        img = Image.open(name)
        img1 = img.transpose(Image.FLIP_LEFT_RIGHT)
        # If the image isn't exist, save it!
        exist_, imgTuple = isExist(img1)
        if exist_:  
            main_printSkip("Rotate", name, imgTuple, len(imgList)-imgNewCount)
        else:
            img1.save("./All/" + str(prefix) + '.' + str(imgCount) + ".jpg")
            #print "Image ", name, "\t: ", imgTuple
            imgCount += 1
            imgNewCount += 1
    print "size: ", len(imgSet)
              
# Remove the whole rest Image
rmAll()

# Make orange images into folder
orange_name_list = listdir("./orange")
main(1, orange_name_list)

# Make apple images into folder
apple_name_list = listdir("./apple")
main(0, apple_name_list)