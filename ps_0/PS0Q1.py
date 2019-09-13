#!/usr/bin/python3

"""PS0Q1.py: Solution to Problem 1(B), sub-question 4.
"""

# Importing Modules
import matplotlib.pyplot as plt
import numpy as np

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

def create_matrix(size=(100,100)):
    """Creates a matrix of given size with non-constant values.
    Parameters
    ----------
    size : tuple(int)
        size of the matrix (rows, cols)
    Returns
    -------
    matrix : np.array
        matrix of given size with non-constant values
    """
    return np.random.rand(size[0], size[1])    

def save_matrix(matrix, file_name = "inputAPS0Q1.npy"):
    """Saves a matrix in npy file
    Parameters
    ----------
    matrix : np.array
        array containing the matrix
    file_name: str (optional)
        File name to store into
    """
    np.save(file_name, matrix)

def load_matrix(file_name = "inputAPS0Q1.npy"):
    """Loads a matrix from npy file
    Parameters
    ----------
    file_name: str (optional)
        Name of the file
    Returns
    -------
    matrix : np.array
    """
    return np.load(file_name)

def generate_intensities(matrix):
    """Generate sorted intensities given a matrix.
    Parameters
    ----------
    matrix : np.array
        The input matrix
    Returns
    -------
    intensity : np.array
        Intensity list of input matrix
    """
    return np.sort(matrix, axis=None)[::-1]

def plot_intensity(intensity):
    """Plots intensity
    Parameters
    ----------
    intensity : np.array
        Matrix of intensity
    """
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    font = {'family' : 'serif', 'size'   : 22}
    plt.rc('font', **font)
    plt.plot(intensity, 'k-')
    # Labels
    plt.xlabel(r'Pixel')
    plt.ylabel(r'Intensities')
    plt.title(r'Intensity Plot')
    # Display plot
    plt.show()

def flatten_image(matrix):
    """Flattens the image to 1D
    Parameters
    ----------
    matrix : np.array
        The image matrix
    Returns
    -------
    flat : np.array
        Flattened version of matrix
    """
    return matrix.reshape(-1)

def plot_histogram(hist, nos_bins=20):
    """Plots histogram
    Parameters
    ----------
    hist : np.array
        histogram to be plotted
    """
    GRAY_COLOR = (0.5,0.5,0.5)
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    font = {'family' : 'serif', 'size'   : 22}
    plt.rc('font', **font)
    # Plot
    plt.hist(hist, bins=nos_bins, color=GRAY_COLOR, ec='black')
    # Labels
    plt.xlabel(r'Bins')
    plt.ylabel(r'Frequency')
    plt.title(r'Intensity Plot')
    # Display plot
    plt.show()

def crop_third_quadrant(matrix):
    """Crops the third quadrant of an image and saves it.
    Parameters
    ----------
    matrix : np.array
        The image matrix
    Returns
    -------
    crop : np.array
        The cropped image
    """
    height, width = matrix.shape
    crop_height = int(height/2)
    crop_width = int(width/2)
    return matrix[-crop_height:, :crop_width]

def plot_image(matrix, is_gray=True):
    """Plots image using plt.imshow and save value
    Parameters
    ----------
    matrix : np.array
        The image matrix
    is_gray : bool (optional)
        Flag to check if image is grayscale
    """
    # Determining the correct colormap
    if (is_gray):
        cmap_option='Greys'
    else:
        cmap_option=None
    # Settings for LaTeX rendering
    # Please comment if your system does not have a native 
    # LaTeX rendering software
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    font = {'family' : 'serif', 'size'   : 22}
    plt.rc('font', **font)
    # Plot
    plt.imshow(matrix, cmap=cmap_option)
    # Labels
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.title(r'Image')
    # Display plot
    plt.show()

def mean_subracted(matrix):
    """Subtracts mean from an image
    Parameters
    ----------
    matrix : np.array
        The image matrix
    Returns:
    --------
    norm_matrix : np.array
        The mean subtracted image
    """
    mean = np.mean(matrix)
    return matrix-mean

def create_rgb_matrix(matrix, threshold):
    """Creates a RGB matrix of given size with red values 
    where the original matrix intensity is higher than mean else
    with black values
    Parameters
    ----------
    matrix : np.array
        Old BW Image
    threshold : np.float
        Threshold for binarization
    Returns
    -------
    rgb_matrix : np.array
        RGB Matrix
    """
    red_channel = matrix
    red_channel[red_channel>threshold]=1.0
    red_channel[red_channel<=threshold]=0.0
    green_channel = np.zeros(matrix.shape)
    blue_channel = np.zeros(matrix.shape)
    return np.dstack([red_channel, green_channel, blue_channel])
    

if __name__=='__main__':
    # Loading the image
    A = create_matrix()
    save_matrix(A)
    A = load_matrix()
    # Sub-problem (a)
    intensity = generate_intensities(A)
    plot_intensity(intensity)
    # Sub-problem (b)
    hist = flatten_image(A)
    plot_histogram(hist)
    # Sub-problem (c)
    X = crop_third_quadrant(A)
    save_matrix(X, "outputXPS0Q1.npy")
    plot_image(X)
    # Sub-problem (d)
    Y = mean_subracted(A)
    save_matrix(Y, "outputYPS0Q1.npy")
    plot_image(Y)
    # Sub-problem (e)
    Z = create_rgb_matrix(A, threshold=np.mean(A))
    plot_image(Z, is_gray=False)