#!/usr/bin/python3
"""Code to display seam"""

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

def displaySeam(im, seam, type):
    """Displays the seam over an image
    Parameters
    ----------
    im : str
        The input image [jpg] only
    seam : list
        The seam
    type : str
        Horizontal or Vertical seam specifier
    """
    # Reading the image
    img = imageio.imread(im)
    h,w,c = img.shape
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    font = {'family' : 'serif', 'size'   : 22}
    plt.rc('font', **font)
    # Plotting the image
    plt.imshow(img)
    # Uncomment this if matplotlib version <= 2.1
    # plt.hold(True)
    # Finding Seam coordinates
    if type == 'HORIZONTAL':
        xs = [i for i in range(w)]
        ys = seam
    elif type == 'VERTICAL':
        xs = seam
        ys = [i for i in range(h)]
    else:
        return
    # Plotting Seam
    plt.plot(xs, ys, 'r-')
    # Labels
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.title(r'Image')
    # Display Image
    plt.show()


def displayBothSeam(im, seamH, seamV):
    """Displays the seam over an image
    Parameters
    ----------
    im : str
        The input image [jpg] only
    seam : list
        The seam
    """
    # Reading the image
    img = imageio.imread(im)
    h,w,c = img.shape
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    font = {'family' : 'serif', 'size'   : 22}
    plt.rc('font', **font)
    # Plotting the image
    plt.imshow(img)
    # Uncomment this if matplotlib version <= 2.1
    # plt.hold(True)
    # Finding Seam coordinates
    xsh = [i for i in range(w)]
    ysh = seamH
    xsv = seamV
    ysv = [i for i in range(h)]
    # Plotting Seam
    plt.plot(xsh, ysh, 'r-', linewidth=1)
    plt.plot(xsv, ysv, 'r-', linewidth=1)
    # Labels
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.title(r'Image')
    # Display Image
    plt.show()


# Uncomment lines from here for testing
# Test functions
def test_displaySeam():
    image_path = "inputSeamCarvingPrague.jpg"
    im = imageio.imread(image_path)
    energy = energy_image(im)
    # Horizontal
    cum_map_h = cumulative_minimum_energy_map(energy, 'HORIZONTAL')
    # view_heat_map(cum_map_h)
    Lh = find_optimal_horizontal_seam(cum_map_h)
    # displaySeam(image_path, Lh, 'HORIZONTAL')
    # Vertical
    cum_map_v = cumulative_minimum_energy_map(energy, 'VERTICAL')
    # view_heat_map(cum_map_v)
    Lv = find_optimal_vertical_seam(cum_map_v)
    # displaySeam(image_path, Lv, 'VERTICAL')
    # Display both
    displayBothSeam(image_path, Lh, Lv)
    

if __name__=="__main__":
    test_displaySeam()