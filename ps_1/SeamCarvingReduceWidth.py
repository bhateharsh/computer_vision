#!/usr/bin/python3
"""Code to reduce width by 100 pixels"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Importing Local Functions
from energy_image import *
from cumulative_minimum_energy_map import *
from find_optimal_horizontal_seam import *
from reduceWidth import *

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# The image list 
imageList = ['inputSeamCarvingPrague.jpg', 'inputSeamCarvingMall.jpg']
# Select the image (0 or 1)
selectImage = 0
imagePath = imageList[selectImage]
# Loading the image
im = imageio.imread(imagePath)
print (im.shape)
# Computing the energy map
energy = energy_image(im)
# Reducing width
reductionSizeW = 100
newImg = np.copy(im)
newEnergy = np.copy(energy)
for i in tqdm(range(reductionSizeW)):
    newImg, newEnergy = reduceWidth(newImg, newEnergy)
# Displaying Output
view_image(newImg)
print (newImg.shape)
if selectImage == 0:
    plt.imsave('newOutputReduceWidthPrague.png',newImg)
else:
    plt.imsave('outputReduceWidthMall.png',newImg)