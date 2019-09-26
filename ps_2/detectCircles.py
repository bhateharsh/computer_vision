#!/usr/bin/python3
"""Code to detect circles using Circular Hough Transform"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray as convertToGray
from skimage.draw import circle
from skimage.feature import canny
from skimage.filters import gaussian, median

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# Helper function
# Helper Functions
def view_image(im):
    """Displays image
    Parameters
    ----------
    im : np.array
        Input image
    """
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    font = {'family' : 'serif', 'size'   : 22}
    plt.rc('font', **font)
    # Plot
    plt.imshow(im, cmap='Greys')
    # Labels
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.title(r'Image')
    # Display plot
    plt.show()

def detectCircles(im, radius, useGradient):
    """Detects Circles using Hough Transform
    Parameters
    ----------
    im : array_like
        The original image (M * N)
    radius : scalar
        Radius candidate
    useGradient : int
        Toggles the usage of gradient
    Returns
    -------
    centers : array_like
        Possible candidates for circles of radius
    """
    # Setting up constants
    PI = 3.14
    # Make sure the image is grayscale
    assert im.ndim == 2
    # Getting the statistics of image
    h, w = im.shape

    # Computing the edge pixel
    edges = canny(im,
                sigma=1.25, 
                low_threshold=0.3,
                high_threshold=0.7)
    # Getting the index of edge pixel
    y, x = np.nonzero(edges)
    assert len(x)==len(y)
    # Defining the accumulator 
    offset = 4 * radius
    acc = np.zeros((im.shape[0] + offset, im.shape[1] + offset), 
                    dtype=np.float64)
    
    # ***************************************************************
    # Part(e)
    # Uncomment this if you want to use binning techniques in finding
    # number of circles
    nosBinsY = int((im.shape[0]+offset)/1)
    nosBinsX = int((im.shape[1]+offset)/1)
    binWidthY = (im.shape[0]+offset)/nosBinsY
    binWidthX = (im.shape[1]+offset)/nosBinsX
    acc = np.zeros((nosBinsY, nosBinsX), dtype=np.float64)
    # ***************************************************************
    # Computing without gradient
    numericalEdges = edges.astype(np.float64)
    gradY, gradX = np.gradient(numericalEdges)
    gradY = - gradY # Correction for y positioning
    print (np.max(x))
    print (np.max(y))
    # Computing gradient
    nosPoints = len(x)
    for i in range(nosPoints):
        if useGradient == 1:
            dy = gradY[y[i], x[i]]
            dx = gradX[y[i], x[i]]
            gradAngle = np.arctan2(dy,dx)
            thetas = [gradAngle]
        else:
            thetas = np.linspace(0, 2*PI, num=360)
        for theta in thetas:
            fineA = x[i] - radius*np.cos(theta)
            fineB = y[i] + radius*np.sin(theta)
            a = int(np.floor(fineA))    
            b = int(np.floor(fineB))
            # *******************************************************
            # Part (e)
            # Uncomment for using binning
            a = int(np.floor(fineA/binWidthY))
            b = int(np.floor(fineB/binWidthX))
            # *******************************************************
            acc[b,a] += 1
    # Setting theshold to extract candidate circles
    
    # ***************************************************************
    # Part(d)
    # Uncomment this if you want to use post processing in finding
    # number of circles
    # threshold = 0.8 # Setting thresholds to allow other bright circle
    # acc = gaussian(acc, sigma=1.0)
    # ***************************************************************
    threshold = 1.0
    centerY, centerX = np.where(acc>=(threshold*np.max(acc)))
    # ***************************************************************
    # Displaying Results for (b), please uncomment if you do not want
    # to see the result plot and only use the test function
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    font = {'family' : 'serif', 'size'   : 22}
    plt.rc('font', **font)
    # Reading color image for display
    img = imageio.imread("egg.jpg")
    # Set up figures
    fig, axes = plt.subplots(1,2)
    print (axes.shape)
    # Displaying the annotated image
    axes[0].cla()
    axes[0].imshow(img)
    nosCandidates = len(centerX)
    print ("binWidthX = %f, binWidthY = %f"\
        %(binWidthX, binWidthY))
    print (centerX)
    print (centerY)
    for i in range(nosCandidates):
        candidateCircle = plt.Circle((centerX[i], centerY[i]), 
                        radius, color='r', fill=False, linewidth=2.5)
        # **********************************************************
        # Part (e)
        # Uncomment if you want to use binning
        candidateCircle = plt.Circle((int(centerX[i]*binWidthX)
                                    , int(centerY[i]*binWidthY)), 
                        radius, color='r', fill=False, linewidth=2.5)
        # *********************************************************** 
        axes[0].add_artist(candidateCircle)
    axes[0].set_title(r"Annotated Image")
    axes[0].set_xlabel(r'x')
    axes[0].set_ylabel(r'y')
    # Displaying the acc
    accFloat = np.copy(acc)
    accFloat /= float(np.max(accFloat))
    axes[1].imshow(accFloat, cmap='hot')
    axes[1].set_title(r'Accumulator Array')
    axes[1].set_xlabel(r'b')
    axes[1].set_ylabel(r'a')
    # Set super title
    fig.suptitle(r'\textbf{Radius = %d, useGradient = %d}'\
            %(radius, useGradient))
    plt.show()
    # ***************************************************************
    
    
    return [centerY, centerX]

# Test
def test_detectCircles():
    """Function to test_detectCircles()
    """
    img = imageio.imread("egg.jpg")
    grayImg = convertToGray(img)
    radius = 8
    gradientToggle = 1
    centerY, centerX = detectCircles(grayImg, radius, gradientToggle)
    # nosCandidates = len(centerX)
    # fig, ax = plt.subplots(1)
    # ax = plt.gca()
    # ax.cla()
    # ax.imshow(img)
    # circles = []
    # print (nosCandidates)
    # for i in range(nosCandidates):
    #     circles.append(plt.Circle((centerX[i], centerY[i]),
    #                 radius, color='r', fill=False, linewidth=2.5))
    #     ax.add_artist(circles[i])
    # plt.show()

# Comment if not in test mode
if __name__=="__main__":
    test_detectCircles();  

# Good Choices
# Egg
# Gradient:YES, r = 4,8, sigma=1.25, low_thres=0.3, high_thres=0.7
# Gradient:NO, r = 4,8, sigma=1.25, low_thres=0.3, high_thres=0.7
# Jupyter
# r = 30,60, sigma = 1.75, low_thres=0.3, high_thres=0.7
