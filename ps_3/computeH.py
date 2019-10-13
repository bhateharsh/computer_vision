#!/usr/bin/python3
"""Code to get compute homography parameters"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from getCorrespondance import plotAndLog

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# Helper Functions
def normalizeCoords(filename, pts):
    """Normalize coordinates between 0 and 2
    Parameters
    ----------
    filename : str
        The filename
    pts : array
        The array to normalize
    Returns
    -------
    normalizedPts : array
        The normalized array
    """
    img = imageio.imread(filename)
    H,W,_ = img.shape
    normalizedPts = np.copy(pts)
    normalizedPts[0, :] = (normalizedPts[0, :]/float(H))*2.0
    normalizedPts[1, :] = (normalizedPts[1, :]/float(W))*2.0
    return normalizedPts

def computeH(t1, t2):
    """Computes the Homography Matrix, H
    Parameters
    ----------
    t1 : array_like
        2xN dimension numpy array corresponding to 
        image points in image1
    t2 : array_like
        2xN dimension numpy array corresponding to 
        image points in image2
    Returns
    -------
    H : array_like
        The homography matrix [3x3 numpy array]
    """
    assert t1.shape == t2.shape, "Unequal correspondance points in matrix 1 and 2"
    nPts = t2.shape[1]
    # Extracting the coordinates
    X, Y = t1
    XDash, YDash = t2
    P = np.zeros((8,9))
    # Forming the correspondance matrix
    for i in range(nPts):
        p_x = [
            X[i], Y[i], 1, 
            0, 0, 0, 
            -X[i]*XDash[i], -Y[i]*XDash[i], -XDash[i]
            ]
        p_y = [
            0, 0, 0, 
            X[i], Y[i], 1, 
            -X[i]*YDash[i], -Y[i]*YDash[i], -YDash[i]
            ]
        P[2*i+0, :] = p_x
        P[2*i+1, :] = p_y
    # Computing H
    A = np.matmul(np.transpose(P), P)
    eigValues, eigVectors = np.linalg.eig(A)
    minEigValueIndex = np.argmin(eigValues)
    H = eigVectors[:, minEigValueIndex]
    return H.reshape((3,3))
# Test
if __name__=="__main__":
    logCoords = plotAndLog("testImage.png")
    a = normalizeCoords("testImage.png", logCoords)
    print (a)
    logCoords2 = plotAndLog("testImage.png")
    H = computeH(logCoords, logCoords2)
    print (H)
    print ("H.shape = ", H.shape)