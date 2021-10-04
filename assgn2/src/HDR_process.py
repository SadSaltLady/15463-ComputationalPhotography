import os
import math
import cv2 
from skimage import io
from Linearlization import *
import numpy as np
import matplotlib.pyplot as plt
# Develop RAW images 
# dcraw -4 -w -q 3 -o 1 -T exposure2.nef

#1.2 Linearlize rendered images 
#import (just a few)all images

#Metadata
kmax = 16
k = 5
N = 400#200
filepath = "../data/door_stack/"

#import all images into this array
allimages = ImportAllImages(k, filepath)
#generate downsampled copies of images for calculation
downsampled = GetDownsampled(N, allimages)
P = np.size(downsampled[1])
'''number of pixels per channel you sample'''

#initialize matrix A
Aw = 3*P*k+256
Ah = 3*P+256
A = np.zeros([Ah, Aw])

#attempt to fill in A
print(np.shape(downsampled[1]))
print(downsampled[1][9][2])
for kIndex in range(1, len(downsampled)): #per level
    for i in range(0, P): #per sampled pixel
        for chnl in range(0, 3): #per RGB channel
            w = 0

    

