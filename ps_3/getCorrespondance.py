#!/usr/bin/python3
"""Code to get correspondance"""

# Importing Modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

# Global Variable
logCoords = []

def recordClick(event):
    """Log click location
    Parameters
    ----------
    event : matplotlib object
        The input canvas to plot location
    Output
    ------
    logCoords : list
        Log of clicked locations
    """
    global iX, iY
    global logCoords
    iX = event.xdata
    iY =  event.ydata
    logCoords.append((iX,iY))
    print ("(x: %d, y: %d)"%(iX,iY))
    plt.plot(iX,iY, 'gx')
    plt.draw()
    if (len(logCoords) == 25):
        fig.canvas.mpl_disconnect(cid)
    return logCoords

def plotAndLog(filename):
    #     """Function to plot and log image
    #     Parameters
    #     ----------
    #     filename : str
    #         Name of the image file to display
    #     """
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    font = {'family' : 'serif', 'size'   : 22}
    plt.rc('font', **font)
    # Setting up a figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Loading Image
    img = imageio.imread(filename)
    # Plot Image
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', recordClick)
    # Show
    plt.show()

# Test function
def test_plotAndLog():
    """Function to test plot and log"""
    filename = "testImage.png"
    plotAndLog(filename)
    print ("Length of coords: %d"%(len(logCoords)))

# Main
if __name__=="__main__":
    test_plotAndLog()