import os
import math
import cv2
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from numpy.core.fromnumeric import reshape, shape
from numpy.core.numeric import identity
from numpy.lib.function_base import median 
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

from UniversalHelpers import *
from cp_hw2 import *


HDR = readHDR("doorwayTIFF_Log_photon.hdr")
K = 0.3
B = 3
outname = "TONEMAP.hdr"
TMtype = "RGB"

def getIWhite(B, HDRcurly, RGBChannel = False):
    '''RGBChannel: true if evaluate each channel separately, false will the channel data and take average'''
    x, y, _ = np.shape(HDRcurly)
    maxR = np.amax(HDRcurly[:, :, 0:1])
    maxG = np.amax(HDRcurly[:, :, 1:2])
    maxB = np.amax(HDRcurly[:, :, 2:3])

    if (RGBChannel):
        bc_maxR = np.full((x, y), maxR)
        bc_maxG = np.full((x, y), maxG)
        bc_maxB = np.full((x, y), maxB)
        max = np.stack((bc_maxR, bc_maxG, bc_maxB), axis=-1)
    else:
        maxRGB = (maxR + maxB + maxG)
        max = np.full((x, y, 3), maxRGB)
    return B * max
    

def getHDRcurly(K, HDRm, HDR):
    Kstack = np.broadcast_to([K, K, K], np.shape(HDR))

    return  HDR * (Kstack / HDRm)

def getHDRm(HDR, RGBChannel = False):
    x, y, _= np.shape(HDR)
    N = x * y

    #get sum for RGB
    if (RGBChannel):
        sumstep = np.sum(np.log(HDR + eps), axis=0)
        sums = np.sum(sumstep, axis=0)
    else:
        sumstep = np.sum(HDR, axis=0)
        sums = np.log(np.sum(sumstep + eps, axis=0))

    top = sums / N
    RM = np.exp(top)

    return np.broadcast_to(RM, np.shape(HDR))

def ToneMappingRGBMain():
    HDRm = getHDRm(HDR, RGBChannel=True)
    HDRcurly = getHDRcurly(K, HDRm, HDR)
    white = getIWhite(B, HDRcurly, RGBChannel=True)
    top = ((HDRcurly / (white ** 2)) + 1.0) * HDRcurly
    bot = 1.0 + HDRcurly 
    HDRtm = top/bot

    #plotting
    N = 400
    TMsample = HDRtm[::N, ::N].flatten()
    HDRsample = HDR[::N, ::N].flatten()

    plt.plot(HDRsample, TMsample,'o')
    plt.show()

    imgmax = np.nanmax(HDRtm.flatten())
    HDRtm = np.nan_to_num(HDRtm, copy = False, nan = imgmax)
    #get rid of all black pixels
    black = np.sum(HDRtm, axis= -1)
    filter = np.stack((black, black, black), axis= -1)
    HDRtm_filtered = np.where(filter <= 0.0, imgmax, HDRtm)
    return HDRtm_filtered


def ToneMappingXYZMain():    
    w, h, _ = np.shape(HDR)
    HDRXYZ = lRGB2XYZ(HDR)
    X = HDRXYZ[:, :, 0:1].reshape((w,h))
    Y = HDRXYZ[:, :, 1:2].reshape((w,h))
    Z = HDRXYZ[:, :, 2:3].reshape((w,h))

    xx = X / (X + Y + Z + eps)
    yy = Y / (X + Y + Z + eps)
    xyY = np.stack((xx, yy, Y), axis=-1)
    
    funny = HDRXYZ    
    HDRm = getHDRm(xyY)
    #keep the x and y channel
    HDRcurly = getHDRcurly(K, HDRm, xyY)
    white = getIWhite(B, HDRcurly)
    top = ((HDRcurly * (white ** 2)) + 1.0) * HDRcurly
    bot = 1.0 + HDRcurly  
    HDRtm = top/bot

    #recovering X and Z
    newX = HDRtm[:, :, 2:3].reshape((w,h)) / (yy * xx + eps)
    newZ = HDRtm[:, :, 2:3].reshape((w,h)) / (yy - newX - HDRtm[:, :, 2:3].reshape((w,h)) + eps)

    #recover XZ

    #only map the Y channel
    HDRXYZ[:, :, 0:1] = newX.reshape((w,h, 1))
    HDRXYZ[:, :, 1:2] = HDRtm[:, :, 2:3]
    HDRXYZ[:, :, 2:3] = newZ.reshape((w,h, 1))
    HDRRGB = XYZ2lRGB(HDRXYZ)

    N = 400
    TMsample = HDRXYZ[::N, ::N].flatten()
    HDRsample = HDR[::N, ::N].flatten()

    plt.plot(HDRsample, TMsample,'o')
    plt.show()

    imgmax = np.nanmax(HDRRGB.flatten())
    HDRRGB = np.nan_to_num(HDRRGB, copy = False, nan = imgmax)
    HDRRGB = np.where(HDRRGB < 0.0, imgmax, HDRRGB)

    return HDRRGB

if(TMtype == "XYZ"):
    writeHDR(outname, ToneMappingXYZMain())
elif(TMtype == "RGB"):
    writeHDR(outname, ToneMappingRGBMain())
print("I run")

#writeHDR("tonemapped_linear_3.hdr", ToneMappingRGBMain())
