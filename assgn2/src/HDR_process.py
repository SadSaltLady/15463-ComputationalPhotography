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

from UniversalHelpers import *
from cp_hw2 import *



#metadata
kmax = 16
'''number of exposures'''

k = 16 #16
'''currently working exposures'''



filepath = "../data/door_stack_linear/"
LDRfilepath = "../data/door_stack/"
type = ".jpg"
merge = "linear"
outname = "doorwayJPG_linear_guassian.hdr"


def HDRStackMain(fp, ldrfp, type, mergetype, outname):
    #importing
    print("Reading " + str(k) + " Files for HDR...")
    if (type == ".jpg"): 
        linear = ImportAllImages(k, fp, type, isINT=True)
        typemax = jpgMax
        LIN = GetNormalized(linear, typemax)

        LDR_path = ldrfp
        LDR = ImportAllImages(k, LDR_path, ".jpg", isINT=True)
        LDR = GetNormalized(LDR, jpgMax)
    else:
        linear = ImportAllImages(k, fp, ".tiff")
        typemax = tiffMax
        LIN = GetNormalized(linear, typemax)
        LDR = LIN

    '''
    if(mergetype == "log"):
        HDR = LogarithmicMerging(LDR, LIN)
    else:
        HDR = LinearMerging(LDR, LIN)
'''

    HDR = LogarithmicMerging(LDR, LIN, type = "photon")
    writeHDR("doorwayJPG_log_photon.hdr", HDR)
    HDR = LogarithmicMerging(LDR, LIN, type = "tent")
    writeHDR("doorwayJPG_log_tent.hdr", HDR)
    HDR = LogarithmicMerging(LDR, LIN, type = "uniform")
    writeHDR("doorwayJPG_log_uniform.hdr", HDR)
    HDR = LogarithmicMerging(LDR, LIN, type = "guassian")
    writeHDR("doorwayJPG_log_guassian.hdr", HDR)

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
        w = np.where(z1 <= ZMax, weightGuassian(z1), 0)

        #do the actual stacking operation
        topK = w * Linijk / exposure(k) #top of the dividing
        top += topK
        bot += w 
    
    #calculate for per pixel value
    imgHDR = top / bot
    return imgHDR


def LinearMerging1(LDR, lin, type = "tiff"):
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
        w = np.where(z1 <= ZMax, weightPhoton(z1, k), 0)

        #do the actual stacking operation
        topK = w * Linijk / exposure(k) #top of the dividing
        top += topK
        bot += w 
    
    #calculate for per pixel value
    imgHDR = top / bot
    return imgHDR

def LinearMerging2(LDR, lin, type = "tiff"):
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
        w = np.where(z1 <= ZMax, weightUniform(z1), 0)

        #do the actual stacking operation
        topK = w * Linijk / exposure(k) #top of the dividing
        top += topK
        bot += w 
    
    #calculate for per pixel value
    imgHDR = top / bot
    return imgHDR

def LinearMerging3(LDR, lin, type = "tiff"):
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
        w = np.where(z1 <= ZMax, weightTent(z1), 0)

        #do the actual stacking operation
        topK = w * Linijk / exposure(k) #top of the dividing
        top += topK
        bot += w 
    
    #calculate for per pixel value
    imgHDR = top / bot
    return imgHDR

def LogarithmicMerging(LDR, lin, type = "photon"):
    top = np.zeros(np.shape(LDR[1]))
    bot = np.zeros(np.shape(LDR[1]))

    for k in range(1, len(LDR)):
        print("currently working on exposure of " + str(k))
        log_tk = np.log(exposure(k))
        LDRijk = LDR[k] #idea: multiple passes and then combine them
        Linijk = lin[k]
        #construct weight matrix
        z1 = np.where(LDRijk >= Zmin, LDRijk, 0)
        if (type == "photon"):
            w = np.where(z1 <= ZMax, weightPhoton(z1, k), 0)
        elif(type == "uniform"):
            w = np.where(z1 <= ZMax, weightUniform(z1), 0)
        elif(type == "tent"):
            w = np.where(z1 <= ZMax, weightTent(z1), 0)
        elif(type == "guassian"):
            w = np.where(z1 <= ZMax, weightGuassian(z1), 0)
        #w = np.where(z1 <= ZMax, weightOptimal(z1, k), 0)

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
    imgmax = np.nanmax(imgHDR.flatten())
    imgHDR = np.nan_to_num(imgHDR, copy = False, nan = imgmax)

    return imgHDR

HDRStackMain(filepath, LDRfilepath, type, merge, outname)