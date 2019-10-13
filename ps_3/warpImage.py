#!/usr/bin/python3
"""Code to get compute homography parameters"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from getCorrespondance import plotAndLog
from computeH import computeH

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

def warpImage(inputIm, refIm, H):
    """Warps Image
    Parameters
    ----------
    inputIm : array_like
        The input image
    refIm : array_like
        The reference image
    H : array
        The homography matrix
    Returns
    -------
    warpIm : array
        The warped image
    mergeIm : array
        The merged image
    """
    
