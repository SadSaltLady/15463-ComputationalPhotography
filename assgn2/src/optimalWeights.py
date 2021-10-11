import os
import math
import cv2 
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

from UniversalHelpers import *

filepath = "../data/my_stack3_tiff/"
k = 11
allfiles = ImportAllImages(k, filepath, ".tiff")
darkframe = np.double(io.imread("darkframe.tiff"))
3.01164905e+01
shutter = [0,1/30, 1/15,1/8, 1/4, 1/1.6, 1.3, 2.5, 5,10,20,40]

for k in range(1, k+1):
    weight = exposure(k) / shutter[k] * darkframe
    subtracted = allfiles[k] - weight
    name = "exposure" + str(k) + ".tiff"
    io.imsave(name, subtracted)



