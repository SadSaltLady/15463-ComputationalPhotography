import os
from matplotlib.colors import Normalize
from skimage import io
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from skimage.color.colorconv import rgb2hsv
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

#FINAL GRAYWORLD

#GrayWorld
def WBGrayWorld(imR, imG, imB):
    #find per channel average
    avgR = np.mean(imR)
    avgG = np.mean(imG)
    avgB = np.mean(imB)
    #apply transformation matrix(weight rbchannels)
    WBimR = imR * (avgG/avgR)
    WBimG = imG 
    WBimB = imB * (avgG/avgB)
    #stack them here (or not)
    #im_rgb = np.dstack((WBimR, WBimG, WBimB))
    return (WBimR, WBimG, WBimB)

def WBWhiteWorld(imR, imG, imB):
    #find per channel max
    maxR = np.amax(imR)
    maxG = np.amax(imG)
    maxB = np.amax(imB)
    #apply transformation matrix(weight rbchannels)
    WBimR = imR * (maxG/maxR)
    WBimG = imG
    WBimB = imB * (maxG/maxB)
    #stack them here (or not)
    #im_rgb = np.dstack((WBimR, WBimG, WBimB))
    #but I guess the brightest spots are everywhere on the image, why would 
    #this function make sense at all? 
    return (WBimR, WBimG, WBimB)

def WBCameraScale(imR, imG, imB):
    #color scale from camera
    rScale = 2.393118
    gScale = 1.0
    bScale = 1.223981
    #Q: if i multiply them doesn't this exceed 1?
    WBimR = imR * rScale
    WBimG = imG * gScale 
    WBimB = imB * bScale
    #stack them here (or not)
    #im_rgb = np.dstack((WBimR, WBimG, WBimB))
    return (WBimR, WBimG, WBimB)

#helper function that zips arrays together
def countList(lst1, lst2):
    return np.array([[i, j] for i, j in zip(lst1, lst2)]).ravel()

#Bayer Pattern
def PatternHelper(BayerType, init):
    #try each one, and see which image looks the best
    #only take one green pixel
    imR = []
    imG = [] #top row green pixel
    imB = []
    imGG = [] #bot row green pixel
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
    
    #make the green pixel array
    imGreen = np.zeros([init.shape[0],init.shape[1]//2])
    imGreen[::2] = imG #first row
    imGreen[1::2] = imGG #second
    return imR, imGreen, imB
    #return WBGrayWorld(imR, imG, imB, imGG)
    


init = ProcessInitial()
imRR, imGG, imBB = PatternHelper("rggb", init)
imR, imG, imB = WBCameraScale(imRR, imGG, imBB)
#now interpolate the images
print(init.shape)
width = init.shape[1]
height = init.shape[0]
#Red channel
xRed = np.arange(0, width, 2)
yRed = np.arange(0, height, 2)
fRed = interpolate.interp2d(xRed, yRed, imR, kind='linear')
imRedChannel = fRed(range(width), range(height))
print("red dimension")
print(imRedChannel.shape)
#Green Channel
#added i%2 to account for odd number pixel images

''' 
########## this overflows and idk what to do with it lol
xGreen = np.array([[j for j in range(i%2, width+i%2, 2)]for i in range(height)])
print("xgreen")
print(xGreen.shape)
yGreen = np.array([[i for j in range(imG.shape[1])] for i in range(height)])
print("ygreen")
print(yGreen.shape)
print("green")
print(imG.shape)    
fGreen = interpolate.interp2d(xGreen.flatten(), yGreen.flatten(), imG, kind='linear')
imGreenChannel = fGreen(range(width), range(height))
'''
xGreen = np.arange(0, width, 2) #PROBLEM
yGreen = np.arange(0, height)
fGreen = interpolate.interp2d(xGreen, yGreen, imG, kind='linear')
imGreenChannel = fGreen(range(width), range(height))
#Blue Channel
xBlue = np.arange(1, width, 2)
yBlue = np.arange(1, height, 2)
print(len(yBlue))
fBlue = interpolate.interp2d(xBlue, yBlue, imB, kind='linear')
imBlueChannel = fBlue(range(width), range(height))



mXYZCam = np.array(
    [
        [6988,-1384,-714],
        [-5631,13410,2447],
        [-1485,2204,7318]
    ]
)*0.001

mRGBXYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ]
)

mRGBCam = np.matmul(mXYZCam, mRGBXYZ)
#normalize it
sumOfRows = mRGBCam.sum(axis=1)
normalizedmRGBCam = mRGBCam / sumOfRows[:, np.newaxis]
inverseM = np.linalg.inv(normalizedmRGBCam)

rgb_camera = np.dstack((imRedChannel, imGreenChannel, imBlueChannel))
plt.imshow(rgb_camera)
plt.show()


def mult(m):
    return np.matmul(inverseM,m)
why = np.apply_along_axis(mult,2,rgb_camera)
print(why.shape)
plt.imshow(why)
plt.show()
'''
rgb_test = np.dstack((imRedChannel, imGreenChannel, imBlueChannel))
print(np.shape(rgb_test))
print (np.shape(init)) #shape = (4016, 6016)
plt.imshow(rgb_test)
plt.show()


'''