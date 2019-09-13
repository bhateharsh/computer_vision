#!/usr/bin/python3
"""Code to reduce height by 100 pixels"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from tqdm import tqdm

# Importing Local Functions
from energy_image import *
from reduceHeight import *
from reduceWidth import *

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# The image list 
imageList = ['testImage1.jpg', 'testImage2.jpg', 'testImage3.jpg']
# Select the image (0,1 or 2)
selectImage = 2
imagePath = imageList[selectImage]
# Loading the image
im = imageio.imread(imagePath)
h, w, c = im.shape
# Computing the energy map
energy = energy_image(im)
# Reducing width
reductionSizeH = 100
reductionSizeW = 125
newShape = (h-reductionSizeH, w-reductionSizeW)
# Seam Carving
newImg = np.copy(im)
newEnergy = np.copy(energy)
# for i in tqdm(range(reductionSizeW)):
#     newImg, newEnergy = reduceWidth(newImg, newEnergy)
for i in tqdm(range(reductionSizeH)):
    newImg, newEnergy = reduceHeight(newImg, newEnergy)
for i in tqdm(range(reductionSizeW)):
    newImg, newEnergy = reduceWidth(newImg, newEnergy)
print (newImg.shape)
print (im.shape)
# Traditional resizing
resizedIm = imresize(im, newShape)
# Displaying Outputs
opt1 = r'Height Reduction followed by Width Reduction'
opt2 = r'Widht Reduction followed by Height Reduction'
# Plotting 
# Settings for LaTeX rendering
# Please comment if your system does not have a native 
# LaTeX rendering software
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
font = {'family' : 'serif', 'size'   : 12}
plt.rc('font', **font)
# Plots
fig, axs = plt.subplots(3)
# Subplot 1
axs[0].imshow(im)
axs[0].set_title(r'Original Image')
# Subplot 2
axs[1].imshow(newImg)
axs[1].set_title(r'Seam Carving')
# # Subplot 3
axs[2].imshow(resizedIm)
axs[2].set_title(r'Image Resizing') 

# Labels
fig.suptitle(r'\textbf{Seam Carving vs Image Resizing}')
# Display plot
plt.show()