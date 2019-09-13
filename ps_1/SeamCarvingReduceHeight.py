#!/usr/bin/python3
"""Code to reduce height by 100 pixels"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Importing Local Functions
from energy_image import *
from cumulative_minimum_energy_map import *
from find_optimal_vertical_seam import *
from reduceHeight import *

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
reductionSizeH = 100
newImg = np.copy(im)
newEnergy = np.copy(energy)
for i in tqdm(range(reductionSizeH)):
    newImg, newEnergy = reduceHeight(newImg, newEnergy)
# Displaying Output
view_image(newImg)
print (newImg.shape)
if selectImage == 0:
    plt.imsave('newOutputReduceHeightPrague.png',newImg)
else:
    plt.imsave('outputReduceHeightMall.png',newImg)