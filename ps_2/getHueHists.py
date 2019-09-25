#!/usr/bin/python3
"""Code to generate Hue Histograms"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from skimage.color import rgb2hsv as convertToHSV

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# Helper functions
def view_histogram(hist, nosBin = None, rangeVal = None):
    """Displays image
    Parameters
    ----------
    hist : array_like
        The histogram
    nosBin : int
        Number of bins
    rangeVal : list
        Range of histogram
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
    plt.bar(np.arange(len(hist)), hist, color='k')
    # Labels
    plt.xlabel(r'Hue Values ($\textrm{in Degree}^\circ$)')
    plt.ylabel(r'Number of Pixels')
    plt.title(r'Histogram of Image')
    # Display plot
    plt.show()

# HueHist
def getHueHists(im, k):
    """Generates histogram for hue and quantized hue
    Parameters
    ----------
    im : array_like (np.uint8)
        The original image of shape M * N * 3 in RGB format
    k : int
        Number of cluster centers
    Returns
    -------
        histEqual : array_like
            The hue histogram of image
        histClustered : array_like
            The hue histogram of quantized image
    """
    h, w, c = im.shape
    # Convert image to HSV
    hsvIm = convertToHSV(im)
    hueIm = hsvIm[:,:,0]
    # Quantize HSV
    # Generating a pixel matrix (reshaped image)
    pixelImg = np.reshape(hueIm, (-1, 1))
    # Performing k-means clustering
    meanHues, pixelLabel = kmeans2(pixelImg, k)
    # Reshaping to get quantized image back
    quantizedIm = np.zeros((h,w))
    labelID = 0
    for i in range(h):
        for j in range(w):
            quantizedIm[i][j] = meanHues[pixelLabel[labelID]]
            labelID += 1
    # Generate histogram
    histEqual, _ = np.histogram(hueIm.ravel(), 
                    bins=360)  
    histClustered, _ = np.histogram(quantizedIm.ravel(),
                    bins=360)
    # Display plot
    plt.show()
    return [histEqual,histClustered]  

# Test
def test_getHueHists():
    """Function to test getHueHists()
    """
    # Toggle comments to choose test image
    # image_path = "testImages/testImage1.png"
    image_path = "testImages/testImage2.jpg"
    # Read Image
    im = imageio.imread(image_path)
    # Set number of cluster centers
    k = 20
    # Running the function
    hist1,hist2 = getHueHists(im, k)
    # Plotting histogram
    view_histogram(hist1)
    view_histogram(hist2)

# Comment if test is not required
# Main
if __name__=="__main__":
    test_getHueHists();