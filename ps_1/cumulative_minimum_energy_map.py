#!/usr/bin/python3
"""Code to compute the cumulative minimum energy map"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from energy_image import *

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# Helper Functions
def view_heat_map(im):
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
    plt.imshow(im)# cmap='plasma')
    # Labels
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.title(r'Image')
    # Display plot
    plt.show()

# Minimum Energy map
def cumulative_minimum_energy_map(energyImage, seamDirection):
    """Computes the cumulative minimum energy map along a given direction
    Parameters
    ----------
    energyImage : np.array (np.double)
        The energy of each pixel in image computed by an energy function
    seamDirection : str
        Specifies direction of seam (VERTICAL OR HORIZONTAL)
    Returns
    -------
    cumulative_energy : np.array (np.double)
        The cumulative minimum energy map along a direction.
    """
    # Setting Seam Direction
    if seamDirection == 'VERTICAL':
        cumulative_energy = np.copy(energyImage)
    elif seamDirection == 'HORIZONTAL':
        cumulative_energy = np.transpose(np.copy(energyImage))
    else:
        return 
    # Extracting energy info
    h, w = cumulative_energy.shape
    # Generating cumulative minimum energy map
    for row in range(1, h):
        prev_row = row - 1
        for col in range(1, w-1):
             cumulative_energy[row, col] = cumulative_energy[row, col] + np.min(cumulative_energy[prev_row, col-1:col+2])
        # Edge Cases
        cumulative_energy[row, 0] = cumulative_energy[row, 0] + np.min(cumulative_energy[prev_row, 0:2])
        cumulative_energy[row, w-1] = cumulative_energy[row, w-1] + np.min(cumulative_energy[prev_row, w-2:w])
    # Returning 
    if seamDirection == 'HORIZONTAL':
        return np.transpose(cumulative_energy)
    return cumulative_energy

# Uncomment lines from here for testing
# Test functions
def test_cumulative_minimum_energy_map():
    image_path = "inputSeamCarvingPrague.jpg"
    im = imageio.imread(image_path)
    energy = energy_image(im)
    cum_map = cumulative_minimum_energy_map(energy, 'HORIZONTAL')
    # cum_map = cumulative_minimum_energy_map(energy, 'VERTICAL')
    view_heat_map(cum_map)
    
if __name__=="__main__":
    test_cumulative_minimum_energy_map();