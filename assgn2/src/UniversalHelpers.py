import os
import math
from numpy.core.fromnumeric import var
from skimage import io
import numpy as np

eps = 0.000000000000001 #14 zeros

tiffMax = 2**16 - 1
'''max pixel value in tiff file size'''

jpgMax = 255

varAdditive = 1.32563420e+06 * 7779.020122813273/tiffMax
g = 3.01164905e+01 - 7779.020122813273/tiffMax


def exposure(k):
    return 1/2048 * (2**(k-1))

    
def GetDownsampledof1(N, imgs):
    '''downsample the images in imgs by the factor of N'''
    downsampled = [np.empty(0)]
    for i in range(1, len(imgs)):
        singleDownsample = imgs[i][::N, ::N]
        downsampled.append(singleDownsample)
    return downsampled

def ImportAllImages(k, path, type, isINT = False, name = "exposure", custom = 0):
    '''reads tiffs with exposure values  <= k from path with specifided imaage type
    (ie, exposure_k_.tiff),
    returns an array where each index is an image reference, 
    [0] is empty such that index = k '''
    
    allImages = [np.empty(0)]
    for i in range(1 + custom, k+1 + custom):
        filename = path + name + str(i) + type
        if(isINT):
            readFile = np.uint8(io.imread(filename))
        else:
            readFile = np.double(io.imread(filename))
        allImages.append(readFile)
    
    return allImages


def GetNormalized(imgs, max):
    '''divide each pixel by the max'''
    normalized = [np.empty(0)]
    for i in range(1, len(imgs)):
        singleNormalized = imgs[i]/max
        normalized.append(singleNormalized)
    return normalized


#weighting functions
Zmin = 0.05
ZMax = 0.95
#this was not as good of an idea as I thought, only used in linearlization
def Weighting(Z, type, k = 0):
    '''Different Weighting functions for Z(0-255), switch weighting functions by changing 
    input 'type'; possible values include: 
    uniform, tent, guassian, photon (k/exposure value required)
    '''
    Z = Z/255
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



#Weighting Fast
def weightUniform(Z):
    '''assumes Z between 0-1'''
    return 1

def weightTent(Z):
    '''assumes Z between 0-1'''
    #min (Z, 1-Z)
    oneminus = 1.0 - Z
    return np.where(Z > oneminus, oneminus, Z)

def weightGuassian(Z):
    '''assumes Z between 0-1'''
    return math.e**(-4 * ((Z - 0.5)**2) / (0.5**2))

def weightPhoton(Z,k):
    '''assumes Z between 0-1'''
    return 1/2048 * (2**(k-1))

def weightCustomPhoton(i):
    exposure = [0, 1/15, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0, 8.0, 15.0]
    return exposure[i]

def weightOptimal(Z, k):
    return (exposure(k) ** 2)/(g*Z + varAdditive)
