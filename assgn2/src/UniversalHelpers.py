import os
import math
from skimage import io
import numpy as np

def exposure(k):
    return 1/2048 * (2**(k-1))

def ImportAllImages(k, path, type):
    '''reads tiffs with exposure values  <= k from path with specifided imaage type
    (ie, exposure_k_.tiff),
    returns an array where each index is an image reference, 
    [0] is empty such that index = k '''
    
    allImages = [np.empty(0)]
    for i in range(1, k+1):
        filename = path + "exposure" + str(i) + type
        readFile = np.double(io.imread(filename))
        allImages.append(readFile)
    
    return allImages

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

