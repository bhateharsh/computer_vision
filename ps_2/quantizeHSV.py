#!/usr/bin/python3
"""Code to quantize HSV image using k-means clustering"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from skimage import img_as_ubyte as convertToInt
from skimage.color import rgb2hsv as convertToHSV
from skimage.color import hsv2rgb as convertToRGB


# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

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

# QuantizedHSV Function
def quantizeHSV(origImg, k):
    """Quantizes image in the 3D HSV space using k-means clustering
    Parameters
    ----------
    origImg : array_like (np.uint8)
        The original image of shape M * N * 3 in RGB format
    k : int
        Number of centers for clustering
    Returns
    -------
    outputImg : array_like (np.uint8)
        Quantized image of shape M * N * 3 in RGB
    meanColors : array_like
        Array containing location of the k centers (k*3)
    """
    # Extracting the shape information for the image
    h, w, c = origImg.shape
    # Running sanity check
    assert c == 3, "Image is not color!"
    # Converting image to hsv
    hsvImg = convertToHSV(origImg)
    print (hsvImg.dtype)
    hueImg = hsvImg[:, :, 0]
    satImg = hsvImg[:, :, 1]
    valImg = hsvImg[:, :, 2]
    # Generating a pixel matrix (reshaped image)
    pixelImg = np.reshape(hueImg, (-1, 1))
    # Performing k-means clustering
    meanColors, pixelLabel = kmeans2(pixelImg, k)
    # Reshaping to get quantized image back
    quantizedImg = np.zeros((h,w))
    labelID = 0
    for i in range(h):
        for j in range(w):
            quantizedImg[i][j] = meanColors[pixelLabel[labelID]]
            labelID += 1
    # Adding to the colorspace
    outputHSV = np.dstack((quantizedImg, satImg, valImg))
    outputImg = convertToInt(convertToRGB(outputHSV))
    return [outputImg, meanColors]

# Test
def test_quantizeHSV():
    """Function to test quantizeHSV
    """
    # Toggle comments to choose test image
    # image_path = "testImages/testImage1.png"
    # image_path = "testImages/testImage2.jpg"
    # Read Image
    im = imageio.imread(image_path)
    # Qunatize Image
    k = 20
    outputImg, meanColors = quantizeHSV(im, k)
    # Generate output and report
    view_image(outputImg)
    print ("REPORT")
    print ("Cluster Centers, k = %d"%(k))
    print ("Input Image Shape: %d * %d * %d"\
        %(im.shape[0], im.shape[1], im.shape[2]))
    print ("Data type of input Image:", im.dtype)
    print ("Shape of meanColors: %d * %d"\
        %(meanColors.shape[0], meanColors.shape[1]))
    print ("Output Image Shape: %d * %d * %d"\
        %(outputImg.shape[0],outputImg.shape[1],outputImg.shape[2]))
    print ("Data type of output Image: ", outputImg.dtype)

# Comment if test is not required
# Main
if __name__=="__main__":
    test_quantizeHSV();