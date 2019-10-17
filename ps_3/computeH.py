#!/usr/bin/python3
"""Code to get compute homography parameters"""

# Importing Modules
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from getCorrespondance import plotAndLog
from tabulate import tabulate
from skimage.transform import ProjectiveTransform
from matplotlib import lines
import math

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

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
    rows = t2.shape[1]
    # Extracting the coordinates
    xs, ys = t1
    xd, yd = t2
    # Generating L
    L = []
    for i in range(rows):
        row1 = []
        row2 = []
        # Point
        point = [xs[i], ys[i], 1]
        # Computing row 1
        row1 = row1 + point
        row1 = row1 + [0, 0, 0]
        row1 = row1 + list(np.multiply(point, -xd[i]))
        # Computing row 2
        row2 = row2 + [0, 0, 0]
        row2 = row2 + point
        row2 = row2 + list(np.multiply(point, -yd[i]))
        # Appending
        L.append(row1)
        L.append(row2)
    # Finding the SVD
    L = np.array(L)
    A = np.matmul(np.transpose(L), L)
    S,U,V = np.linalg.svd(A)
    H = V[-1, :]
    return H.reshape(3,3)
    
# -- TEST FUNCTIONS ----------------------------------------------------------

# Verify computeH
def verify_computeH():
    """Function to verify computeH by plotting"""
    inputIm = imageio.imread("crop1.jpg")
    refIm = imageio.imread("crop2.jpg")
    # Generate New Canvas
    hIn, wIn, c = inputIm.shape
    hRef, wRef, _ = refIm.shape
    hNew = max(hIn,hRef)
    wNew = wIn+wRef
    newCanvas = np.zeros([hNew, wNew, c], dtype=np.uint8)
    # Plot Params
    offsetX = wIn; offsetY = hIn
    # Loading correspondence points
    t1 = np.load("cc1.npy").T
    t2 = np.load("cc2.npy").T
    # Computing H
    H = computeH(t1, t2)
    # Copying t1 to point format
    x1,y1 = t1
    pt1 = np.array(list(zip(list(x1), list(y1))))
    # Finding point 2 via H
    pt2 = []; x2 = []; y2 = []
    for point in pt1:
        point = np.append(point, 1)
        xT, yT, scale = np.matmul(H, point)
        xT = int(xT/scale)
        yT = int(yT/scale)
        transformedPoint = np.array([xT, yT])
        pt2.append(transformedPoint)
        x2.append(xT)
        y2.append(yT)
    pt2 = np.array(pt2)
    # Plotting the pointsgs[1])
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("Input Image")
    ax2.set_title("Reference Image")
    ax1.imshow(inputIm, aspect="auto")
    ax2.imshow(refIm, aspect="auto")

    for i in range(len(x1)):
        transFigure = fig.transFigure.inverted()
        coord1 = transFigure.transform(ax1.transData.transform([x1[i],y1[i]]))
        coord2 = transFigure.transform(ax2.transData.transform([x2[i],y2[i]]))
        line = lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                            transform=fig.transFigure)
        fig.lines.append(line)
    ax1.scatter(x1,y1,color='r', marker='x')
    ax2.scatter(x2,y2,color='r', marker='x')
    plt.show()

# Test 
def test_computeH():
    """Function to test computeH
    """
    # Read Points from cc1
    t1 = np.load("cc1.npy").T
    t2 = np.load("cc2.npy").T
    # Converting to form for Cv2
    x,y = t1
    pt1 = np.array(list(zip(list(x), list(y))))
    x,y = t2
    pt2 = np.array(list(zip(list(x), list(y))))
    # Running functions
    H_indigenous = computeH(t1, t2)
    H_developed = cv2.findHomography(pt1, pt2)
    transform = ProjectiveTransform()
    result = transform.estimate(t1.T, t2.T)
    if (result):
        H_skimage = transform.params
    # plotting
    x,y = t2
    img = imageio.imread("crop2.jpg")
    plt.imshow(img)
    plt.scatter(x,y)
    plt.show()
    # Print output
    # H_indigenous = H_indigenous/H_indigenous[2,2]
    headers = ["1", "2", "3"]
    table1 = tabulate(H_indigenous, headers, tablefmt="fancy_grid")
    table2 = tabulate(H_developed[0], headers, tablefmt="fancy_grid")
    table3 = tabulate(H_skimage, headers, tablefmt="fancy_grid")
    print ("Indigenous Matrix")
    print (table1)
    print ("Developed Matrix")
    print (table2)
    print ("Skimage Matrix")
    print (table3)
    # Print 
    norm = np.linalg.norm(H_indigenous)
    print ("Norm of H_indegenous = ", norm)

    norm = np.linalg.norm(H_developed[0])
    print ("Norm of H_developed = ", norm)

    norm = np.linalg.norm(H_skimage)
    print ("Norm of H_skimage = ", norm)
    # Testing output
    Xs = []
    Ys = []
    for point in pt1:
        xs, ys = point
        zs = 1
        h1 = H_skimage[0,:]
        h2 = H_skimage[1,:]
        h3 = H_skimage[1,:]
        # print (H_skimage@np.array([xs,ys,zs]).T)
        out = np.array(H_developed) @ np.array([xs,ys,zs]).T
        print (out)
        xd = xd/scale
        yd = yd/scale
        Xs.append(xd)
        Ys.append(yd)
    a = np.array([Xs, Ys]).T
    headers = ["x", "y"]
    table4 = tabulate(a.T, headers, tablefmt="fancy_grid")
    table5 = tabulate(t2, headers, tablefmt="fancy_grid")
    print ("Homography Out")
    print(table4)
    print ("Actual Out")
    print (table5)
    

# Test
if __name__=="__main__":
    # test_computeH()
    verify_computeH()