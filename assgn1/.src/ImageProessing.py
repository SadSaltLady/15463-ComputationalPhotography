import os
import math
from matplotlib.colors import Normalize
from numpy.lib.shape_base import dstack
from skimage import io
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from skimage.color.colorconv import rgb2hsv

def main():
    #READING IMAGE AND LINEARLIZING---------------------------------------------
    init = ProcessInitial()
    ##DETERMINING PATTERN-------------------------------------------------------
    #TIP: switch the string to try different encoding
    imRR, imGG, imBB, imGGTop, imGGBot = PatternHelper("rggb", init)

    #WHITE BALANCING------------------------------------------------------------
    #options include: 
    # WBGrayWorld, WBWhiteWorld, WBWhiteWorldManual, WBCameraScale
    imR, imG, imB = WBGrayWorld(imRR, imGG, imBB)

    #DEMOSAIC-------------------------------------------------------------------
    #now interpolate the images
    width = init.shape[1]
    height = init.shape[0]
    imRedChannel, imGreenChannel, imBlueChannel = Demosaicing (imR, imGGTop, imGGBot, imB, width, height)
    #COLOR SPACE CORRECTION-----------------------------------------------------
    inverseM = GetInverseM()
    rgb_camera = np.dstack((imRedChannel, imGreenChannel, imBlueChannel))
    #can I still keep my channels separatae at this point - yes
    CSCRed = imRedChannel * inverseM[0][0] + imGreenChannel * inverseM[0][1] + imBlueChannel*inverseM[0][2]
    CSCGreen = imRedChannel * inverseM[1][0] + imGreenChannel * inverseM[1][1] + imBlueChannel*inverseM[1][2]
    CSCBlue = imRedChannel * inverseM[2][0] + imGreenChannel * inverseM[2][1] + imBlueChannel*inverseM[2][2]

    #Brightness adjustment and gamma encoding-----------------------------------
    LinearScalingConst = 1.55
    gammaRed, gammaGreen, gammaBlue = linearScaling(LinearScalingConst, CSCRed, CSCGreen, CSCBlue)
    rgb_camera = np.dstack((gammaRed, gammaGreen, gammaBlue))

    #Display Image--------------------------------------------------------------
    plt.imshow(rgb_camera)     
    plt.show()
    #save the picture
    io.imsave('WBGrayWorld_final.png', rgb_camera)

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


#WhiteBalancing-----------------------------------------------------------------
#for all white balancing functions: 
#INPUT: Red, Green, Blue, second row Green pixels in 2D array
#OUTPUT: R, G, B pixel 2D arrays after white balancing

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

def WBWhiteWorldManual(imR, imG, imB):
    #collected values manually and just putting them here as constants
    #Currently using: Clouds
    #White area: clouds
    '''
    maxR = 0.431
    maxG = 0.998
    maxB = 0.69
    '''
    #White area: stairs

    maxR = 0.312
    maxG = 0.788
    maxB = 0.527

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
#--------------------------------------------------------------------------
#helper function that zips arrays together
def countList(lst1, lst2):
    return np.array([[i, j] for i, j in zip(lst1, lst2)]).ravel()

#Bayer Pattern------------------------------------------------------------
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
    return imR, imGreen, imB, imG, imGG
    #return WBGrayWorld(imR, imG, imB, imGG)
    
#NOTE: demosaicing is assumping rggb layout -----------------------------------
def Demosaicing (imR, imGTop, imGBot, imB, width, height):
    #Red channel
    xRed = np.arange(0, width, 2)
    yRed = np.arange(0, height, 2)
    fRed = interpolate.interp2d(xRed, yRed, imR, kind='linear')
    imRedChannel = fRed(range(width), range(height))
    #Blue Channel
    xBlue = np.arange(1, width, 2)
    yBlue = np.arange(1, height, 2)
    fBlue = interpolate.interp2d(xBlue, yBlue, imB, kind='linear')
    imBlueChannel = fBlue(range(width), range(height))
    #Top Green Chanel
    xGTop = np.arange(0, width, 2)
    yGTop = np.arange(1, height, 2)
    fGTop = interpolate.interp2d(xGTop, yGTop, imGTop, kind = 'linear')
    imGreenTop = fGTop(range(width), range(height))
    #Bot Green Channel
    xGBot = np.arange(1, width, 2)
    yGBot = np.arange(0, height, 2)
    fGBot = interpolate.interp2d(xGBot,yGBot, imGBot, kind = 'linear')
    imGreenBot = fGBot(range(width), range(height))
    #Combine and take average 
    imGreenChannel = (imGreenTop + imGreenBot) / 2.
    return imRedChannel, imGreenChannel, imBlueChannel

#COLOR SPACE ---------------------------------------------------------------
#Calculates the color space transformation matrix
def GetInverseM():
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
    return np.linalg.inv(normalizedmRGBCam)

#Gamma Encoding--------------------------------------------------------------
#helper
def GamEncoding(x):
    if (x <= 0.0031308):
        return 12.92 * x
    else:
        return ((1 + 0.055) * math.pow(x, 1./2.4)) - 0.055


def linearScaling(scale, CSCRed, CSCGreen, CSCBlue):
    gammaRed = CSCRed * scale
    gammaGreen = CSCGreen * scale
    gammaBlue = CSCBlue * scale
    gammaRed = np.vectorize(GamEncoding)(gammaRed)
    gammaBlue = np.vectorize(GamEncoding)(gammaBlue)
    gammaGreen = np.vectorize(GamEncoding)(gammaGreen)
    gammaRed = np.clip(gammaRed, 0.,1.)
    gammaGreen = np.clip(gammaGreen, 0., 1.)
    gammaBlue = np.clip(gammaBlue, 0. , 1.)
    return (gammaRed, gammaGreen, gammaBlue)

main()
