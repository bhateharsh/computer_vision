#!/usr/bin/python3
"""Code to reduce height by 1 pixel"""

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

def reduceHeight(im, energyImage):
    """Reduced the heigh of an image by remove the optimal horizontal seam
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
    im_trans = np.transpose(im, axes=[1,0,2])
    h,w,c = im_trans.shape
    # Compute the cumulative horizontal energy of the image
    cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, 'HORIZONTAL')
    # Compute the optimal horizontal seam
    seam = find_optimal_horizontal_seam(cumulativeEnergyMap)
    # Generate Index list to delete
    xs = seam
    ys = [i for i in range(h)]
    # Removing seam from image
    mask = np.ones(im_trans.shape, dtype = bool)
    mask[ys, xs, :] = False
    reducedColorImage = im_trans[mask]
    reducedImage = np.transpose(reducedColorImage.reshape(h, w-1, c), axes=[1,0,2])
    # Computing new energy map
    reducedEnergyImage = energy_image(reducedImage)
    # Return the value
    return [reducedImage, reducedEnergyImage]

# Uncomment lines from here for testing
# Test functions
# def test_reduceHeight():
#     image_path = "inputSeamCarvingPrague.jpg"
#     im = imageio.imread(image_path)
#     energy = energy_image(im)
#     new_im, new_energy = reduceHeight(im, energy)
#     view_image(im)
#     view_image(new_im)
#     view_image(energy)
# if __name__=="__main__":
#     test_reduceHeight();