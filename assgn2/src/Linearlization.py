
import os
import math
import cv2 
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

# Develop RAW images 
# dcraw -4 -w -q 3 -o 1 -T exposure2.nef

def exposure(k):
    return 1/2048 * (2**(k-1))

def ImportAllImages(k, path):
    '''reads tiffs with exposure values  <= k from path(exposure_k_.tiff),
    returns an array where each index is an image reference, 
    [0] is empty such that index = k '''
    
    allImages = [np.empty(0)]
    for i in range(1, k+1):
        filename = path + "exposure" + str(i) + ".jpg"
        readFile = np.double(io.imread(filename))
        allImages.append(readFile)
    
    return allImages

def GetDownsampled(N, imgs):
    '''downsample the images in imgs by the factor of N'''
    downsampled = [np.empty(0)]
    for i in range(1, len(imgs)):
        singleDownsample = imgs[i][::N, ::N]/255
        downsampled.append(singleDownsample)
    return downsampled

def Weighting(Z, type, k = 0):
    '''Different Weighting functions for Z, switch weighting functions by changing 
    input 'type'; possible values include: 
    uniform, tent, guassian, photon (k/exposure value required)
    '''
    Zmin = 0.05
    ZMax = 0.95
    if (Z <= ZMax and Z >= Zmin):
        if (type == "uniform"): 
            return 1
        elif (type == "tent"):
            return min(Z, 1 - Z)
        elif (type == "guassian"):
            return math.e**(-4 * ((Z - 0.5)**2) / (0.5**2))
        elif (type == "photon"):
            if (k <= 0):
                print("need to provide valid k value")
                assert(1 == 2)
            return exposure(k)
        else: 
            print("please provide valid 'type")
            assert(1 == 2)
    else:
        return 0


