import os
from skimage import io
import numpy as np
import scipy
import matplotlib.pyplot as plt
#task 1
def ProcessInitial():
    #some constants
    black = 150.
    white = 4095.
    #read image
    imgr = io.imread("campus.tiff") 
    #linearlize: 
    #max = 4095, black = 150, white = 4095
    imgr = np.double(imgr)
    imgrLinear = (imgr - black) / (white - black)
    imgrLinear = np.clip(imgrLinear, 0.0, 1.0)
    return imgrLinear


#WhiteBalancing
#for all white balancing functions: 
#INPUT: Red, Green, Blue, second row Green pixels in 2D array
#OUTPUT: R, G, B pixel 2D arrays after white balancing
#GrayWorld
def WBGrayWorld(imR, imG, imB, imGG):
    pixelLen = len(imR[0]) * len(imR)
    #find per channel average
    avgR = sum(sum(imR)) / pixelLen
    avgG = (sum(np.append(imG,imGG))) / (pixelLen * 2.)
    avgB = sum(sum(imB)) / pixelLen
    #apply transformation matrix(weight rbchannels)
    WBimR = imR * (avgG/avgR)
    WBimG = np.add(imG, imGG) / 2. #naive: taking the average of the two
    WBimB = imB * (avgG/avgB)
    #stack them here (or not)
    #im_rgb = np.dstack((WBimR, WBimG, WBimB))
    return (WBimR, WBimG, WBimB)

def WBWhiteWorld(imR, imG, imB, imGG):
    #find per channel max
    maxR = np.amax(imR)
    maxG = np.amax(np.append(imG,imGG))
    maxB = np.amax(imB)
    print("debug:" + str(maxG))
    print("debug:" + str(maxB))
    #apply transformation matrix(weight rbchannels)
    WBimR = imR * (maxG/maxR)
    WBimG = np.add(imG, imGG) / 2. #naive: taking the average of the two
    WBimB = imB * (maxG/maxB)
    #stack them here (or not)
    #im_rgb = np.dstack((WBimR, WBimG, WBimB))
    #but I guess the brightest spots are everywhere on the image, why would 
    #this function make sense at all? 
    return (WBimR, WBimG, WBimB)

def WBCameraScale(imR, imG, imB, imGG):
    #color scale from camera
    rScale = 2.393118
    gScale = 1.0
    bScale = 1.223981
    #Q: if i multiply them doesn't this exceed 1?
    WBimR = imR * rScale
    WBimG = np.add(imG, imGG) * gScale / 2.
    WBimB = imB * bScale
    #stack them here (or not)
    #im_rgb = np.dstack((WBimR, WBimG, WBimB))
    return (WBimR, WBimG, WBimB)


#Bayer Pattern
def PatternHelper(BayerType, init):
    initFlat = init.flatten() #makes indexing a little easier
    #try each one, and see which image looks the best
    #only take one green pixel
    imR = []
    imG = []
    imB = []
    imGG = []
    imGreen = []
    print("debug")
    print(initFlat)
    if BayerType == "grbg":
        imR = init[0::2, 1::2]
        imG = init[0::2, 0::2]
        imB = init[1::2, 0::2]
        imGG = init[1::2, 1::2]

    elif BayerType == "rggb":
        imR = init[0::2, 0::2]
        imG = init[0::2, 1::2]
        imB = init[1::2, 1::2]
        imGG = init[1::2, 0::2]
    elif BayerType == "bggr":
        imR = init[1::2, 1::2]
        imG = init[0::2, 1::2]
        imB = init[0::2, 0::2]
        imGG = init[1::2, 0::2]
    elif BayerType == "gbrg":
        imR = init[1::2, 0::2]
        imG = init[0::2, 0::2]
        imB = init[0::2, 1::2]
        imGG = init[1::2, 0::2]
    
    return WBGrayWorld(imR, imG, imB, imGG)
    


init = ProcessInitial()
rgb_test = PatternHelper("rggb", init)

'''

'''
print(np.shape(rgb_test))
print (np.shape(init)) #shape = (4016, 6016)
plt.imshow(rgb_test)
plt.show()

array1 = [1, 2, 3, 4]
array2 = [4, 3, 2, 1]
array3 = np.add(array1, array2)
print(array3)
#print(np.dtype(doublep))