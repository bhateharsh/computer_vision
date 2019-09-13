#!/usr/bin/python3
"""Code to compute the Energy of an image based on L-1 norm of gradients
"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import color
from skimage import filters

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# Helper functions
def check_image(im):
    """Checks the dimensions and datatype of image
    Parameters
    ----------
    im : np.array (uint8)
        Input Array
    """
    dims = im.shape
    assert len(dims)==3, "Input dimension mismatch"
    assert dims[2]==3, "Input Image of depth 3"
    assert np.issubdtype(im.dtype, np.uint8), "Incorrect data type"    

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

def view_all_image(im1, im2):
    """Displays Image
    Parameters
    ----------
    im1 : np.array
        Original Energy image
    im2 : np.array
        New Enerfy Image
    """
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18) 
    font = {'family' : 'serif', 'size'   : 18}
    plt.rc('font', **font)
    # Plot
    fig, axs = plt.subplots(1,2)
    # Subplot 1
    axs[0].imshow(im1, cmap="Greys")
    axs[0].set_title(r'Original Energy Image \\  (Sobel Filter, L-1 Norm)')
    # Subplot 2
    axs[1].imshow(im2, cmap="Greys")
    axs[1].set_title(r'New Energy Image \\ (Gaussian Blur, Scharr Filter, L-2 Norm)')
    # Labels
    fig.suptitle(r'\textbf{Comparison of different Image Energy}')
    # Display plot
    plt.show()

# Energy Function
def new_energy_image(im):
    """Computes the energy of an image.
    Parameters
    ----------
    im : np.array (uint8)
        The original image of size M*N*3
    Output
    ------
    energy : np.array (double)
        The energy of the image
    """
    # Check image
    check_image(im)
    # Convert to grayscale
    gray_im = color.rgb2gray(im)
    gray_im = gray_im.astype(np.double)
    # Using gaussian blur to remove fine details
    # Computing Gradients
    im_y = ndimage.sobel(gray_im, axis=0)
    im_x = ndimage.sobel(gray_im, axis=1)
    # Returning L-2 norm
    return np.sqrt(np.power(im_x, 2) + np.power(im_y,2))
    
# Energy Function
def energy_image(im):
    """Computes the energy of an image.
    Parameters
    ----------
    im : np.array (uint8)
        The original image of size M*N*3
    Output
    ------
    energy : np.array (double)
        The energy of the image
    """
    # Check image
    check_image(im)
    # Convert to grayscale
    gray_im = color.rgb2gray(im)
    gray_im = gray_im.astype(np.double)
    # Using gaussian blur to remove fine details
    gray_im = ndimage.gaussian_filter(gray_im, sigma=1)
    # Uncomment for Alternate gradients
    im_y = filters.scharr_v(gray_im)
    im_x = filters.scharr_h(gray_im)
    # Computing Gradients
    # Uncomment the following line to use alternate energy function (L-2 norm)
    # Returning Value
    return np.absolute(im_x) + np.absolute(im_y)


# Uncomment lines from here for testing
# Test functions
def test_check_image():
    image_path = "inputSeamCarvingPrague.jpg"
    im = imageio.imread(image_path)
    check_image(im)

def test_energy_image():
    image_path = "inputSeamCarvingPrague.jpg"
    im = imageio.imread(image_path)
    energy = energy_image(im)
    view_image(energy)
    # Uncomment to run both energy functions
    newEnergy = new_energy_image(im)
    view_all_image(energy, newEnergy)
    
# Main Function
if __name__=="__main__":
    test_energy_image();