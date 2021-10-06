import os
import math
import cv2
from numpy.core.fromnumeric import reshape, shape
from numpy.core.numeric import identity 
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

from Linearlization import *
from UniversalHelpers import *
from cp_hw2 import *



#metadata
kmax = 16
'''number of exposures'''

k = 16
'''currently working exposures'''

tiffMax = 2**16 - 1
'''max pixel value in tiff file size'''

def HDRStackMain():
    tiff_path = "../data/door_stack_tiff/"
    #import stacked TIFF images
    print("Reading " + str(k) + " Tiff Files for HDR...")
    tiff = ImportAllImages(k, tiff_path, ".tiff")
    tiff_normalized = GetNormalized(tiff, tiffMax)

    HDR = LogarithmicMerging(tiff_normalized, tiff_normalized)
    writeHDR("testing16_log.hdr", HDR)
    print("image succeessfully generated")


def LinearMerging(LDR, lin, type = "tiff"):
    top = np.zeros(np.shape(LDR[1]))
    bot = np.zeros(np.shape(LDR[1]))

    if (type == "tiff"):
        lin = LDR

    for k in range(1, len(LDR)):
        print("currently working on exposure of " + str(k))
        LDRijk = LDR[k] #idea: multiple passes and then combine them
        Linijk = lin[k]
        #construct weight matrix
        z1 = np.where(LDRijk >= Zmin, LDRijk, 0)
        w = np.where(z1 <= ZMax, weightPhoton(z1,k), 0)

        #do the actual stacking operation
        topK = w * Linijk / exposure(k) #top of the dividing
        top += topK
        bot += w 
    
    #calculate for per pixel value
    imgHDR = top / bot
    return imgHDR

def LogarithmicMerging(LDR, lin, type = "tiff"):
    top = np.zeros(np.shape(LDR[1]))
    bot = np.zeros(np.shape(LDR[1]))
    eps = 0.000000000000001 #14 zeros
    if (type == "tiff"):
        lin = LDR

    for k in range(1, len(LDR)):
        print("currently working on exposure of " + str(k))
        log_tk = np.log(exposure(k))
        LDRijk = LDR[k] #idea: multiple passes and then combine them
        Linijk = lin[k]
        #construct weight matrix
        z1 = np.where(LDRijk >= Zmin, LDRijk, 0)
        w = np.where(z1 <= ZMax, weightPhoton(z1,k), 0)

        #construct the log matrix
        logs = np.log(Linijk + eps) - log_tk

        #do the actual stacking operation
        topK = w * logs 
        top += topK
        bot += w 

    #get rid of weird values inside of the bottom
    #calculate for per pixel value
    imgHDR = top / bot
    imgHDR = np.exp(imgHDR)
    return imgHDR

HDRStackMain()