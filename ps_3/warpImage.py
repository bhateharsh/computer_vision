#!/usr/bin/python3
"""Code to get compute homography parameters"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from getCorrespondance import plotAndLog
from computeH import computeH
from tabulate import tabulate
from skimage.transform import ProjectiveTransform, warp
import pdb

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
    # Dimension
    hIn,wIn,C = inputIm.shape
    hRef, wRef, C = refIm.shape
    # -- Warping -------------------------------------------------------------
    # Find Bounding Box
    xs = []
    ys = []
    for j in range(hIn):
        for i in range(wIn):
            x, y, scale = np.matmul(H,np.array([[i],[j],[1]]))
            x = x/scale
            y = y/scale
            xs.append(x)
            ys.append(y)
    # Find max dims
    minX = min(xs)
    maxX = max(xs)
    minY = min(ys)
    maxY = max(ys)
    # Computing new Image dims
    if (minY <= 0):
        hNew = int(np.ceil(maxY-minY)+1)
    if (minX <= 0):
        wNew = int(np.ceil(maxX-minX)+1)
    if (minY > 0):
        hNew = int(np.ceil(maxY-minY)+1)    
    if (minX > 0):
        wNew = int(np.ceil(maxX-minX)+1)
    newCanvas = np.zeros([hNew, wNew, C], dtype=np.uint8)
    # Running
    for j in range(hIn):
        for i in range(wIn):
            x, y, scale = np.matmul(H,np.array([[i],[j],[1]]))
            xMin = int(np.floor (x/scale - minX))
            yMin = int(np.floor(y/scale - minY))
            # Splattering
            newCanvas[yMin,xMin] = inputIm[j,i]
            newCanvas[yMin+1,xMin] = inputIm[j,i]
            newCanvas[yMin,xMin+1] = inputIm[j,i]
            newCanvas[yMin+1,xMin+1] = inputIm[j,i]
            xs.append(x)
            ys.append(y)
    # Inverse
    invH = np.linalg.inv(H)
    for j in range(hNew):
        for i in range(wNew):
            shiftedX = int(i + minX)
            shiftedY = int(j + minY)
            xs, ys, scale = np.matmul(invH, 
                                      np.array([[shiftedX],[shiftedY],[1]]))
            xs = int(xs/scale)
            ys = int(ys/scale)
            if (xs < wIn) and (ys < hIn) and (xs>=0) and (ys>=0):
                newCanvas[j,i] = inputIm[ys, xs]
    # -- Merge ---------------------------------------------------------------
    # Dimension Selection Algorithm
    if (minY >= 0):
        hMerge = hNew + int(abs(minY))
    else:
        lastPoint = hRef + abs(minY)
        if (lastPoint>hNew):
            hMerge=lastPoint
        else:
            hMerge=hNew
    if (minX >= 0):
        wMerge = wNew + int(abs(minX))
    else:
        lastPoint = wRef + abs(minX)
        if (lastPoint>wNew):
            wMerge=lastPoint
        else:
            wMerge=wNew
    if (hRef > hMerge):
        hMerge = hRef
    if (wRef > wMerge):
        wMerge = wRef
    mergedCanvas = np.zeros([hMerge, wMerge, C], dtype=np.uint8)
    # Generating Offsets
    if minY <= 0:
        offsetY = 0
    else:
        offsetY = int(minY)
    if minX <= 0:
        offsetX = 0
    else:
        offsetX = int(minX)
    mergedCanvas[offsetY-int(minY):hRef-int(minY)+offsetY, 
                 offsetX-int(minX):wRef-int(minX)+offsetX,:] = refIm
    yInd,xInd,_ = np.nonzero(newCanvas)
    mergedCanvas[yInd+offsetY, xInd+offsetX,:] = newCanvas[yInd, xInd, :] 
    return [newCanvas, mergedCanvas]

# Comment if not in test
# Test Function
def test_warpIm(testID=0):
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18) 
    font = {'family' : 'serif', 'size'   : 18}
    plt.rc('font', **font)
    # Testing
    if testID==0:
        # Reading Images
        inputIm = imageio.imread("crop1.jpg")
        refIm = imageio.imread("crop2.jpg")
        # Extracting Correspondance Points
        t1 = np.load("cc1.npy").T
        t2 = np.load("cc2.npy").T
        # Compute homography matrix
        H = computeH(t1, t2)
        # Compute Warp
        warpIm, mergeIm = warpImage(inputIm, refIm, H)
        # TEST ONLY 
        # Scipy 
        # transform = ProjectiveTransform()
        # transform.estimate(t2.T, t1.T)
        # H = transform.params
        # testImg = warp(inputIm, H, clip=False)
        # plt.figure()
        # plt.title("Scipy")
        # plt.imshow(testImg)
        # plt.show()
    elif testID==1:
        # Reading images
        inputIm = imageio.imread("wdc1.jpg")
        refIm = imageio.imread("wdc2.jpg")
        # Extracting Correspondance Points
        t1 = plotAndLog("wdc1.jpg")
        t2 = plotAndLog("wdc2.jpg")
        # Saving Points
        print ("Saving Points")
        print ("t1.shape = ", t1.shape)
        np.save("points1.npy", t1)
        np.save("points2.npy", t2)
        # Compute homography matrix
        H = computeH(t1, t2)
        # H = np.load("bhag.npy")
        table = tabulate(H, tablefmt="fancy_grid")
        print ("Homography Matrix:")
        print (table)
        # Compute Warp
        warpIm, mergeIm = warpImage(inputIm, refIm, H)
    elif testID==2:
        # Reading images
        inputIm = imageio.imread("mosaicTestIn.jpeg")
        refIm = imageio.imread("mosaicTestRef.jpeg")
        # Extracting Correspondance Points
        t1 = plotAndLog("mosaicTestIn.jpeg")
        t2 = plotAndLog("mosaicTestRef.jpeg")
        # Saving Points
        print ("Saving Points")
        print ("t1.shape = ", t1.shape)
        np.save("mosaic1.npy", t1)
        np.save("mosaic2.npy", t2)
        # Compute homography matrix
        H = computeH(t1, t2)
        table = tabulate(H, tablefmt="fancy_grid")
        print ("Homography Matrix: ")
        print (table)
        # Compute Warp
        warpIm, mergeIm = warpImage(inputIm, refIm, H)
    else:
        # Reading images
        inputIm = imageio.imread("goodBoy.jpg")
        refIm = imageio.imread("poster.jpg")
        # Extracting Correspondance Points
        h,w,_ = inputIm.shape
        maxX = w-1
        maxY = h-1
        minX = 0
        minY = 0  
        x = np.array([minX, maxX, maxX, minX])
        y = np.array([minY, minY, maxY, maxY])
        t1 = np.array([x,y])
        t2 = plotAndLog("poster.jpg")
        # Compute homography matrix
        H = computeH(t1, t2)
        table = tabulate(H, tablefmt="fancy_grid")
        print ("Homography Matrix: ")
        print (table)
        # Compute Warp
        warpIm, mergeIm = warpImage(inputIm, refIm, H)
        # Save
        imageio.imsave("posterBoy.png", mergeIm)
    # Plot Graphs
    plt.figure(1)
    plt.title("Warped Image")
    plt.imshow(warpIm)
    plt.show()
    plt.figure(2)
    plt.title("Merged Image")
    plt.imshow(mergeIm)
    plt.show()
    
            
# Main
if __name__=="__main__":
    test_warpIm(2)