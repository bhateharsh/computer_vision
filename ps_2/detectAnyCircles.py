#!/usr/bin/python3
"""Code to detect any circle using Circular Hough Transform"""

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

def detectCircles(im, useGradient):
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
    # Setting up radii bank
    radii = np.linspace(0,8, num=4)
    radii = radii[1:]
    allCircles = dict()
    # Make sure the image is grayscale
    assert im.ndim == 2
    # Getting the statistics of image
    h, w = im.shape

    # Computing the edge pixel
    edges = canny(im,
                sigma=1.0, 
                low_threshold=0.3,
                high_threshold=0.7)
    # Getting the index of edge pixel
    y, x = np.nonzero(edges)
    assert len(x)==len(y)
    # Computing without gradient
    numericalEdges = edges.astype(np.float64)
    gradY, gradX = np.gradient(numericalEdges)
    gradY = - gradY # Correction for y positioning
    # Computing gradient
    nosPoints = len(x)
    for radius in radii:
        # Defining the accumulator 
        offset = int(2 * radius)
        acc = np.zeros((im.shape[0] + offset, im.shape[1] + offset), 
                        dtype=np.float64)
        for i in range(nosPoints):
            if useGradient == 1:
                dy = gradY[y[i], x[i]]
                dx = gradX[y[i], x[i]]
                gradAngle = np.arctan2(dy,dx)
                thetas = np.linspace(gradAngle - (PI/4), 
                            gradAngle + (PI/4), 
                            num=180)
            else:
                thetas = np.linspace(0, 2*PI, num=360)
            for theta in thetas:
                fineA = x[i] - radius*np.cos(theta)
                fineB = y[i] + radius*np.sin(theta)
                a = int(np.round(fineA))    
                b = int(np.round(fineB))
                acc[b,a] += 1
        # Setting theshold to extract candidate circles
        threshold = 0.9
        centerY, centerX = np.where(acc>=(threshold*np.max(acc)))
        allCircles.update({radius:[centerY, centerX]})
    # ***************************************************************
    # Displaying Results
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
    # Displaying the annotated image
    axes[0].cla()
    axes[0].imshow(img)
    for key, value in allCircles.items():
        centerY = value[0]
        centerX = value[1]
        nosCandidates = len(centerX)
        for i in range(nosCandidates):
            candidateCircle = plt.Circle((centerX[i], centerY[i]), 
                            key, color='r', fill=False, linewidth=2.5)
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
    fig.suptitle(r'Multiple Circles of different Radii (useGradient = %d)'\
            %(useGradient))
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
    centerY, centerX = detectCircles(grayImg, gradientToggle)

# Comment if not in test mode
if __name__=="__main__":
    test_detectCircles();  

# Good Choices
# Egg
# Gradient:YES, r = 4,8, sigma=1.25, low_thres=0.3, high_thres=0.7
# Gradient:NO, r = 4,8, sigma=1.25, low_thres=0.3, high_thres=0.7
# Jupyter
# r = 30,60, sigma = 1.75, low_thres=0.3, high_thres=0.7
