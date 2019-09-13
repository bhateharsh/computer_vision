#!/usr/bin/python3
"""Code to reduce width by 1 pixel"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np

# Importing Local Functions
from energy_image import *
from cumulative_minimum_energy_map import *
from find_optimal_horizontal_seam import *
from find_optimal_vertical_seam import *

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

def reduceWidth(im, energyImage):
    """Reduced the width of an image by remove the optimal vertical seam
    Parameters
    ----------
    im : np.array (np.uint8)
        The image matrix
    energyImage :np.array (np.float64/double)
        Energy of the image
    Returns
    -------
    reducedImage : np.array (np.uint8)
        The reduced width image
    reducedEnergyImage : np.array (np.float64/double)
        The reduced width energy image
    """
    h,w,c = im.shape
    # Compute the cumulative vertical energy of the image
    cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, 'VERTICAL')
    # Compute the optimal vertical seam
    seam = find_optimal_vertical_seam(cumulativeEnergyMap)
    # Generate Index list to delete
    xs = seam
    ys = [i for i in range(h)]
    # Removing seam from image
    mask = np.ones(im.shape, dtype = bool)
    mask[ys, xs, :] = False
    reducedColorImage = im[mask]
    reducedImage = reducedColorImage.reshape(h, w-1, c)
    # Computing new energy map
    reducedEnergyImage = energy_image(reducedImage)
    # Return the value
    return [reducedImage, reducedEnergyImage]

# Uncomment lines from here for testing
# Test functions
# def test_reduceWidth():
#     image_path = "inputSeamCarvingPrague.jpg"
#     im = imageio.imread(image_path)
#     energy = energy_image(im)
#     new_im, new_energy = reduceWidth(im, energy)
#     view_image(im)
#     view_image(new_im)
#     view_image(energy)

# if __name__=="__main__":
#     test_reduceWidth();