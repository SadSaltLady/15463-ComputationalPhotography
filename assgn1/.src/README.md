COMPLETED:
ALL tasks in the handout
Bonus: built camera obscura, more details in pdf and data/cameraObscura


Python File:  ImageProcessing.py

Function and functionalities: 
main(): 
    The main program, comment show different steps of the process. 

    Parameters you can modify: 
    -PatternHelper (line 16)->> intake string determines pixel pattern
    -WB______ (line 21)->>Different white balancing algorithms. 
        WBGrayWorld, WBWhiteWorld, WBWhiteWorldManual, WBCameraScale
    -LinearScalingConst (line 37)->> brightness scaling constant

ProcessInitial():
    Contains process for: Python initials, Linearization

    Parameters you can modify: 
    Location of the source file (?)

##White Balancing Family:
    Contains Process for white balancing

    Input: R, G, B as 2D pixel arrays
    Output: White balanced R, G, B 2D pixel arrays

    WBGrayWorld(imR, imG, imB)
    WBWhiteWorld(imR, imG, imB)
    WBCameraScale(imR, imG, imB)
    WBWhiteWorldManual(imR, imG, imB)
        **default to balancing based on cloud white, uncomment things to make it balance to the stair white. Sorry for this unfriendly way of controlling code, but I wanted to keep the White balancing functions inputs consistent. 

PatternHelper(BayerType, init)
    Seperates a big array into R, G, B channels based on input Bayer pattern. 

    Input: BayerType - string describing the bayer pattern
           init - mosaiced 2D image array
    Output: 
        imR - 2D array corresponding to the red pixels
        imGreen - 2D array corresponding to *ALL* the green pixels
        imB - 2D array corresponding to the blue pixels
        imG - 2D array corresponding to the green pixels on the first row of the bayer pattern
        imGG - 2D array corresponding to the green pixels on the second row of the bayer pattern


Demosaicing (imR, imGTop, imGBot, imB, width, height)
    Performs demosaicing by performing linear interpolation based on the 3 channels and image size

    #NOTE: the demosaicing algorithm assumes the pattern is rggb

    Input: 
        Red, FirstRowGreen, SecondRowGreen, Blue channels as 2D arrays
        width, height - image widtha and height
    
    Output: 
        R, G, B 2D pixel arrays size of width * height, interpolated from data provided per channel in the input

GetInverseM()
    Calculates the matrix used to correct color space based on slides

GamEncoding(x)
    helper function(vectorized) used to perform gamma encoding on values 

linearScaling(scale, CSCRed, CSCGreen, CSCBlue)
    Performs Gamma encoding and brightness adjustment on RGB channels

    Input: 
        Scale : brightness adjust amount
        RGB channels after Color Space Correction
    
    Output: 
        RGB channels after brightness adjustment and gamma encoding

