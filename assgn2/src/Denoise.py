import os
import math
import re
import cv2
from matplotlib import colors
from numpy.core.fromnumeric import mean, reshape, shape
from numpy.core.numeric import identity
from numpy.lib.function_base import average, median 
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

from UniversalHelpers import *
from cp_hw2 import *

def getDarkFrame():
    N = 50
    path = "../data/noiseCalibration/"
    print("importing images...")
    blackstack = ImportAllImages(N, path, ".tiff", False, "DSC_0", 128)
    print("finding average")

    end = np.mean(blackstack[1:], axis= 0)

    print(end[20][20])
    io.imsave("darkframe_part5.png", end)

def getDarkFrame2():
    f1 = np.double(io.imread("darkframe_part1.tiff"))
    f2 = np.double(io.imread("darkframe_part2.tiff"))
    f3 = np.double(io.imread("darkframe_part3.tiff"))
    f4 = np.double(io.imread("darkframe_part4.tiff"))
    f5 = np.double(io.imread("darkframe_part5.tiff"))

    end = np.mean([f1, f2, f3, f4, f5], axis = 0)
    print(end[20][20])
    io.imsave("darkframe.tiff", end)


def GenerateBase():
    N = 29
    path = "../data/ramp/"
    print("importing images...")
    darkframe = np.double(io.imread("darkframe.tiff"))
    stack = ImportAllImages(N, path, ".tiff", False, "DSC_0", 99)
    print("finding new")
    for i in range(1, N + 1):
        print("for img " + str(i))
        new = stack[i] - darkframe
        newname = "new_" + str(21 + i) + ".tiff"
        io.imsave(newname, new)
        
    print("completed till" )

tiffMax = 2**16 - 1

def plottingHistogram():
    N = 50
    path = "../data/subtracted/"
    readall = ImportAllImages(N, path, ".tiff", False, "new_")

    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    for i in range(1, 51):
        pixel = np.average(readall[i][2000][2000])
        data1.append(pixel)
        pixel = np.average(readall[i][2000][2500])
        data2.append(pixel)
        pixel = np.average(readall[i][2000][3000])
        data3.append(pixel)
        pixel = np.average(readall[i][2000][3500])
        data4.append(pixel)
        pixel = np.average(readall[i][2000][4000])
        data5.append(pixel)
    '''
    plt.plot(range(0, 50), data1)
    plt.plot(range(0, 50), data2)
    plt.plot(range(0, 50), data3)
    plt.plot(range(0, 50), data4)
    plt.show()
    '''

def Variance():
    N = 50
    path = "../data/subtracted/"
    readall = ImportAllImages(N, path, ".tiff", False, "new_")

    print("all images loaded")
    print(readall[4][2000][2000])

    mean = readall[1] / N
    for i in range(2, N + 1):
        print(i)
        mean += readall[i] / N

    print(mean)
    print(np.average(mean.flatten()))

    #mean 
    #vaiance 
    variance = np.zeros(np.shape(readall[1]))
    for i in range(1, N + 1):
        print(i)
        variance += np.square((readall[i] - mean))

    variance = variance / (N - 1)
    variance = np.average(variance, axis=2)


    m = np.rint(np.average(mean, axis=2))
    unique = np.unique(m.flatten())[::100]
    print(len(unique))
    meanVariance = []
    for i in range(0, len(unique)):
        print("processing" + str(i))
        seekMean = unique[i]
        mask = np.where(m == seekMean, 1, 0)
        
        masked = mask * variance
        maskedVar = list(filter(None, masked.flatten()))
        avgVar = np.average(np.array(maskedVar))

        meanVariance.append(avgVar)

    print(np.polyfit(unique, meanVariance, 1, full = True))

    plt.plot(unique, meanVariance, 'o')
    plt.plot(unique, np.poly1d(np.polyfit(unique, meanVariance, 1))(unique))
    plt.show()
    

    return


    print(variance[2000][2000])
    mean = np.average(mean, axis=2)
    variance = np.average(variance, axis=2)
    v = variance.flatten()[::1000]
    m = mean.flatten()[::1000]
    
    #approximate location of the ramp





Variance()
#GenerateBase()
#getDarkFrame2()
#plottingHistogram()
