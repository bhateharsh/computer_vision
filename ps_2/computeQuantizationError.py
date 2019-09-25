#!/usr/bin/python3
"""Code to compute SSD Quantization error"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np

# Importing custom modules
from quantizeHSV import quantizeHSV
from quantizeRGB import quantizeRGB

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# QuantizedHSV Function
def computeQuantizationError(origImg, quantizedImg):
    """Computes the quantization error between original
    and quantized image.
    Parameters
    ----------
    origImg : array_like (np.uint8)
        The original image of shape M * N * 3 in RGB format
    quantizedImg : array_like (np.uint8)
        The quantized image of shape M * N * 3 in RGB format
    Returns
    -------
    error : float
        The error value between the original and quantized image
    """
    # Convert the images to float and normalize
    origImg = np.array(origImg, dtype=np.float64)/255
    quantizedImg = np.array(quantizedImg, dtype=np.float64)/255
    # Compute error
    errorImg = origImg - quantizedImg
    squaredError = np.square(errorImg)
    error = np.sum(squaredError)
    return error

# Test
def test_computeQuantizationError():
    """Function to test computeQuantizationError()
    """
    # Toggle comments to choose test image
    image_path = "testImages/testImage1.png"
    # image_path = "testImages/testImage2.jpg"
    # Read Image
    im = imageio.imread(image_path)
    # Set number of cluster centers
    k = 20
    # Toggle to choose quantization method
    # outputImg, meanColors = quantizeRGB(im, k)
    outputImg, meanColors = quantizeHSV(im, k)
    # Compute Error
    error = computeQuantizationError(im, outputImg)
    # Generate output and report
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
    print ("Error (e) = %.2f"%(error))

# Comment if test is not required
# Main
if __name__=="__main__":
    test_computeQuantizationError();