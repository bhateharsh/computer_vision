#!/usr/bin/python3
"""Code to compute the horizontal optimal seam"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from energy_image import *
from cumulative_minimum_energy_map import *

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# Helper Functions

def find_optimal_horizontal_seam(cumulativeEnergyMap):
    """Computes the optimal horizontal seam
    Parameters
    ----------
    cumulativeEnergyMap : np.array (double)
        The cumulative minimum energy map
    Returns
    -------
    seam_vector : np.array
        Vector containing the column indices of the seam pixels for each column
    """
    # Extracting the shape of the energy map
    h,w = cumulativeEnergyMap.shape
    seam_vector = []
    # Computing the minimum energy map for last row
    seam_vector.append(np.argmin(cumulativeEnergyMap[:, w-1]))
    col = w-2
    # Iterating over every row
    while (col >= 0):
        if seam_vector[-1] == 0:
            seam_vector.append(
                np.argmin(cumulativeEnergyMap[0:2, col]))
        elif seam_vector[-1] == h-1:
            seam_vector.append(
                h 
                - 2 
                + np.argmin(cumulativeEnergyMap[h-2:h, col])
            )
        else:
            seam_vector.append(
                seam_vector[-1] 
                - 1 
                + np.argmin(cumulativeEnergyMap[seam_vector[-1]-1:seam_vector[-1]+2, col])
            )
        col = col - 1
    seam_vector.reverse()
    return seam_vector

# Uncomment lines from here for testing
# Test functions
def test_find_optimal_horizontal_seam():
    image_path = "inputSeamCarvingPrague.jpg"
    im = imageio.imread(image_path)
    energy = energy_image(im)
    cum_map = cumulative_minimum_energy_map(energy, 'HORIZONTAL')
    view_heat_map(cum_map)
    L = find_optimal_horizontal_seam(cum_map)
    print (len(L))
    print (L)
    
if __name__=="__main__":
    test_find_optimal_horizontal_seam()