import os
import math
import cv2
from matplotlib import colors
from numpy.core.fromnumeric import reshape, shape
from numpy.core.numeric import identity 
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

from UniversalHelpers import *
from cp_hw2 import *



def UsedDataCaptureCode():
    '''just keeping this here such that I can look at it later'''
    clicks = 24 * 2 + 1
    plt.imshow(HDR)
    store = plt.ginput(clicks,timeout=120, show_clicks=True)
    plt.show()

def CapturedCoordinateCleanup():
    #note: these follow top left corner, bottom right corner
    #following the order in the illustration on the handout
    captured = [ 
    (3322.756388772518, 1412.027524088815), 
    (3417.5964949029467, 1512.4464599916219), 
    (3478.9636223991065, 1417.6063538611932), 
    (3568.2248987571575, 1515.2358748778108), 
    (3637.9602709118844, 1417.6063538611932), 
    (3724.4321323837457, 1515.2358748778108), 
    (3788.5886747660943, 1420.3957687473821), 
    (3877.8499511241453, 1512.4464599916219), 

    (3308.8093143415726, 1255.8202904622265), 
    (3403.6494204720016, 1350.6603965926552), 
    (3470.5953777405393, 1264.1885351207936), 
    (3565.4354838709683, 1336.7133221617098), 
    (3629.592026253317, 1264.1885351207936), 
    (3724.4321323837457, 1356.2392263650333), 
    (3794.1675045384727, 1264.1885351207936), 
    (3875.060536237956, 1345.081566820277), 

    (3317.17755900014, 1096.8236419494488), 
    (3398.0705906996236, 1197.2425778522556), 
    (3473.3847926267285, 1107.9813014942051), 
    (3559.85665409859, 1183.2955034213105), 
    (3629.592026253317, 1091.2448121770708), 
    (3718.8533026113673, 1188.8743331936885), 
    (3791.3780896522835, 1105.1918866080161), 
    (3869.481706465578, 1191.6637480798777), 

    (3308.8093143415726, 940.6164083228604),
    (3386.912931154867, 1035.4565144532892), 
    (3470.5953777405393, 943.4058232090495), 
    (3554.277824326212, 1035.4565144532892), 
    (3621.2237815947497, 940.6164083228604), 
    (3699.3273984080442, 1029.877684680911), 
    (3785.7992598799055, 951.7740678676167), 
    (3855.5346320346325, 1032.6670995671002), 
    
    (3303.2304845691947, 787.198589582461), 
    (3378.5446864962996, 856.933961737188), 
    (3465.0165479681614, 781.6197598100829), 
    (3543.1201647814555, 868.0916212819443), 
    (3615.6449518223717, 792.7774193548391), 
    (3688.1697388632874, 870.8810361681334), 
    (3769.062770562771, 798.3562491272173), 
    (3300.4410696830055, 630.9913559558725), 

    (3386.912931154867, 630.9913559558725), 
    (3456.6483033095938, 711.8843876553558), 
    (3537.5413350090776, 630.9913559558725), 
    (3612.8555369361825, 714.6738025415449), 
    (3682.5909090909095, 628.2019410696835), 
    (3771.85218544896, 714.6738025415449), 
    (3841.587557603687, 594.7289624354144), 
    (3950.374738165061, 711.8843876553558)
    ]

    #want to floor every coordinate so they can be used for indexing
    capturedIndex = [ (math.floor(x), math.floor(y)) for (x, y) in captured]

    return capturedIndex


def Find24Average(index, HDRimg):
    '''Find the average of colors within the box specified by the index'''
    imgCC = []
    for i in range(0, 48, 2):
        x0, y0 = index[i]
        x1, y1 = index[i + 1]
        count = (x1 - x0) * (y1 - y0)

        accum = np.zeros(np.shape(HDRimg[0][0]))
        for y in range(y0, y1):
            for x in range(x0, x1):
                accum += HDRimg[y][x]
        
        avg = accum / count
        #weird thingy cuz numpy does weird concate things 
        #append the extra 1 to make it homogenous 4 x 1
        imgCC.append([avg[0], avg[1], avg[2], 1.0])
        
    return imgCC
        

def GetTransformedColorChecker():
    '''get the color checker values such that ret[0] = block1, ret[8] = block8, etc'''
    colorchecker = read_colorchecker_gm()
    reshaped = []
    for i in range(0, 6):
        for j in range(0, 4):
            R = colorchecker[0][j][i]
            G = colorchecker[1][j][i]
            B = colorchecker[2][j][i]

            reshaped.append([R, G, B])
    
    return reshaped


def ConstructA(imgRGB):
    A = np.zeros((24 * 3, 12))
    for i in range(0, 24):
            for j in range(0, 4):
                A[i*3 + 0][0 * 4 + j] = imgRGB[i][j]
                A[i*3 + 1][1 * 4 + j] = imgRGB[i][j]
                A[i*3 + 2][2 * 4 + j] = imgRGB[i][j]

    return A


def ColorCorrectionMain(filename):
    capturedIndex = CapturedCoordinateCleanup()
    HDR = readHDR(filename)
    colorchecker = GetTransformedColorChecker()
    #change it into a 4*1 matrix
    colorchecker_img = Find24Average(capturedIndex, HDR) 

    '''compute for x following the equation:
        x = (A^T * A)^(-1)A^T * b'''
    A = ConstructA(colorchecker_img)
    b = np.array(colorchecker).flatten()

    AT = np.transpose(A)
    step = scipy.linalg.inv(np.matmul(AT, A)) #center step so code doesn't get too long
    x = np.matmul(np.matmul(step, AT), b)
    #reshape x to make it easier to do multiplication
    x = x.reshape((3, 4))


    shapex, shapey, shapez = np.shape(HDR)
    #x_broadcast = np.broadcast_to(x, (shapey, 3, 4))

    #make the HDR a homogenous
    HDRHomogenous = np.ones((shapex, shapey, 4))
    HDRHomogenous[:,:,0:3] = HDR
    HDRHomogenous.reshape((shapex, shapey, 4, 1))

    final = np.zeros(np.shape(HDR))

    '''apply the transform onto my HDR image, slow af but like this is the best I got'''
    for i in range(shapex):
        for j in range(shapey):
            final[i][j] = np.matmul(x,HDRHomogenous[i][j])
        
        print(i)

    final = np.where(final >= 0, final, 0)
    writeHDR("colorCorrected_linear.hdr", final)
    #HDRHomogenous[:,:,0:3] = fuck(HDRHomogenous[:,:,0:1], HDRHomogenous[:,:,1:2], HDRHomogenous[:,:,2:3])

def WhiteBalancingMain(filename):
    HDR = readHDR(filename)
    capturedIndex = CapturedCoordinateCleanup()
    colorchecker_img = Find24Average(capturedIndex, HDR) 

    white = colorchecker_img[4]
    maxR = white[0]
    maxG = white[1]
    maxB = white[2]

    #maxRGB = max(max(maxR, maxG), maxB)
    maxRGB = (maxR + maxG + maxB)/3.0

    HDR_R = HDR[:, :, 0:1]
    HDR_G = HDR[:, :, 1:2]
    HDR_B = HDR[:, :, 2:3]

    WBHDR_R, WBHDR_G, WBHDR_B = WBWhiteWorldManual(HDR_R, HDR_G, HDR_B, maxRGB, maxR, maxG, maxB)

    whiteBalancedHDR = np.dstack((WBHDR_R, WBHDR_G, WBHDR_B))
    print("yay")
    writeHDR("whiteBalanced_linear.hdr", whiteBalancedHDR)


def WBWhiteWorldManual(imR, imG, imB, max, maxR, maxG, maxB):
    #collected values manually and just putting them here as constants
    #Currently using: Clouds
    #White area: clouds


    #apply transformation matrix(weight rbchannels)
    WBimR = imR * (max/maxR)
    WBimG = imG * (max/maxG)
    WBimB = imB * (max/maxB)
    #stack them here (or not)
    #im_rgb = np.dstack((WBimR, WBimG, WBimB))
    #but I guess the brightest spots are everywhere on the image, why would 
    #this function make sense at all? 
    return (WBimR, WBimG, WBimB)

WhiteBalancingMain("colorCorrected_linear.hdr")
#ColorCorrectionMain("testing16_linear.hdr")