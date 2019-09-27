#!/usr/bin/python3
"""Code to compute color Quantization"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np

# Importing custom modules
from quantizeHSV import quantizeHSV
from quantizeRGB import quantizeRGB
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# Setting Matplotlib Parameters
# Settings for LaTeX rendering
# Please comment if your system does not have a native 
# LaTeX rendering software
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
font = {'family' : 'serif', 'size'   : 10}
plt.rc('font', **font)
    

# Loading the mail Image
origImg = imageio.imread("fish.jpg")

# Quantizing for low k values, k = 5
k = 5
print ("Computing Quantized Images for k = %d"%(k))

# Computing quantizations
# Computing the RGB quantization
outputRGB, meanColors = quantizeRGB(origImg, k)
# Computing the HSV quantization
outputHSV, meanHues = quantizeHSV(origImg, k)

# Computing Errors
errorRGB = computeQuantizationError(origImg, outputRGB)
errorHSV = computeQuantizationError(origImg, outputHSV)

# Computing hue histograms
histOrg, histQuantized = getHueHists(origImg, k)

# Display Error
print ("Error in RGB Quantization = %.2f"%(errorRGB))
print ("Error in HSV Quantization = %.2f"%(errorHSV))

# Displaying the quantized plots
fig1, axs = plt.subplots(1,3)
# Subplot 1
axs[0].imshow(origImg)
axs[0].set_title(r'Original Image')
# Subplot 2
axs[1].imshow(outputRGB)
axs[1].set_title(r'\small RGB Quantized Image$(e= %.2f)$'%(errorRGB))
# Subplot 3
axs[2].imshow(outputHSV)
axs[2].set_title(r'\small HSV Quantized Image $(e = %.2f)$'%(errorHSV))
# Labels
fig1.suptitle(r'\textbf{Quantization Output} $(k=5)$')
# Display plot
fig1.tight_layout()
fig1.show()

# Display Histogram
fig2, axs = plt.subplots(1,2)
# Subplot 1
axs[0].bar(np.arange(len(histOrg)), histOrg, color='k')
axs[0].set_title(r'Hue Histogram of Original Image')
axs[0].set(xlabel=r'Hue Values', ylabel=r'Number of Pixels')
# Subplot 2
axs[1].bar(np.arange(len(histQuantized)), histQuantized, color='k')
axs[1].set_title(r'Hue Histogram of Quantized Image')
axs[1].set(xlabel=r'Hue Values', ylabel=r'Number of Pixels')
# Labels
fig2.suptitle(r'\textbf{Hue Histogram Comparison}')
fig2.tight_layout()
# Display plot
fig2.show()

plt.show()