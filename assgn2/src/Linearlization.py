
import os
import math
import cv2 
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

from UniversalHelpers import *

# Develop RAW images 
# dcraw -4 -w -q 3 -o 1 -T /assgn2/data/my_stack3_nef/exposure1.nef

def LinearlizeImage(image, c):
    change = lambda x: c[x]    
    x, y, z = np.shape(image)
    #bruh how to python
    image.reshape((x*y*z))
    return change(image).reshape((x, y, z))

def Linearlization_main():
    #Metadata
    n = 256
    '''possible values for a pixel'''
    k = 11
    '''number of exposures used to calculate g'''
    N = 200
    '''downsampling rate'''
    lmd = 20
    '''lambda used in the smoothing/regularization term'''
    wz = 1 
    '''weighting for the smoothing/regularization term'''
    filepath = "../data/my_stack3/"

    #import all images into this array
    allimages = ImportAllImages(k, filepath, ".jpg", True)

    #generate downsampled copies of images for calculation
    downsampled = GetDownsampled(N, allimages)

    P = np.size(downsampled[1])
    '''number of pixels per channel you sample'''
    #initialize matrix A
    print(np.shape(downsampled[1]))
    Ah = P*k+256
    Aw = P+256
    A = np.zeros([Ah, Aw])

    #init Vector B
    B = np.zeros(Ah)

    #constants needed
    sampleH = np.size(downsampled[1], 0)
    sampleW = np.size(downsampled[1], 1)

    #Construct the A matrix
    idxing = 0 
    #filling the data terms
    for kIndex in range(1, len(downsampled)): #per level
        logtk = math.log(exposure(kIndex))
        for i in range(0, sampleH): #per sampled pixel width; axis =0 
            for j in range(0, sampleW): #per sampled pixel height; axis = 1
                for chnl in range(0, 3): #per RGB channel
                    I_ijk = downsampled[kIndex][i][j][chnl] 
                    #note: will always be int >= 0 && <= 255
                    weightI = Weighting(I_ijk, "photon", kIndex)
                    #filling in A 
                    A[idxing][int(I_ijk)] = weightI
                    A[idxing][n + (i*sampleW + j)*3 + chnl] += -weightI #should be +=?
                    #filling in our solution B 
                    B[idxing] = weightI * logtk
                    #indexing
                    idxing += 1

    #setting the middle value to 0
    A[idxing][128] = 1 
    A[idxing][129] = 1 
    A[idxing][130] = 1 

    idxing += 1

    #filling in the smoothing terms
    for i in range(0, 256 - 2):
        A[idxing][i] = lmd * wz
        A[idxing][i + 1] = -2 * lmd * wz
        A[idxing][i + 2] = lmd * wz
        idxing += 1


    #try to solve
    print("hey")
    print(idxing)

    x, residuals, rank, s = np.linalg.lstsq(A, B)
    print(np.shape(x))
    print(Ah)
    print(np.shape(residuals))


    g_values = x[0:256]
    
    print(g_values)
    g_values_exp = np.exp(g_values)
    print(g_values_exp)
    
    
    #attempt to graph ->
    #pixel value on the x axis
    #the corresponding log value -> the first p*k value of X

    input = range(0, 256)
    _ = plt.plot(input, g_values, 'o')
    plt.show()
    return allimages, g_values_exp


def GetDownsampled(N, imgs):
    '''downsample the images in imgs by the factor of N'''
    downsampled = [np.empty(0)]
    for i in range(1, len(imgs)):
        singleDownsample = imgs[i][::N, ::N]
        downsampled.append(singleDownsample)
    return downsampled

images, correction = Linearlization_main()
for i in range(1, len(images)):
    madelinear = LinearlizeImage(images[i], correction)
    name = "exposure" + str(i) + ".jpg"
    io.imsave(name, madelinear)




