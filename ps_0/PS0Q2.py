#!/usr/bin/python3

# Importing Modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imsave

# Authorship Information
__author__ = "Harsh Bhate"
__email__ = "bhate@gatech.edu"

def load_image(file_path="inputPS0Q2.png"):
    """Load image to a matrix
    Parameters
    ----------
    file_path : str (optional)
        The file path to load image
    Returns
    -------
    image : np.array
        A np.array matrix
    """
    return imread(file_path)

def switch_channels(image):
    """Switch the Red and Green channels
    Parameters
    ----------
    image : np.array
        A np.array matrix
    Returns
    -------
    swap_image : np.array
        A swapped np.array matrix
    """
    return np.dstack([image[:,:,1], image[:,:,0], image[:,:,2]])

def save_image(image, file_name):
    """Saves Image to file
    Parameters
    ----------
    image : np.array
        The image matrix
    file_name : str
        Path to save the file
    """
    imsave(file_name, image)

def rgb2gray(image):
    """Converts RGB to grayscale
    Parameters
    ----------
    image : np.array
        The image matrix
    Returns
    -------
    gray_image : np.array
        The grayscale image
    """
    return np.uint8(np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]))

def negative_image(image):
    """Creates an image negative
    Parameters
    ----------
    image : np.array
        The image matrix
    Returns
    -------
    neg_image : np.array
        The negative image
    """
    max_pixel = np.max(image)
    return np.uint8(max_pixel - image)

def mirror_image(image):
    """Creates mirror image
    Parameters
    ----------
    image : np.array
        The image matrix
    Returns
    -------
    flip_image : np.array
        The mirror image
    """
    return image[:, ::-1]

def avg_image(gray_image, mirror_image):
    """Averages image and mirror_image
    Parameters
    ----------
    image : np.array
        The image matrix
    mirror_image : np.array
        The mirror image
    Returns
    -------
    avg_img : np.array
        The average image
    """
    gray_image = gray_image/255.0
    mirror_image = mirror_image/255.0
    return np.uint(((gray_image+mirror_image)/2.0)*255.0)

def generate_noise(size):
    """Generates noise
    Parameters
    ----------
    size : tuple(int)
        The size of noise
    Returns
    -------
    noice : np.array
        The noise matrix
    """
    return np.random.rand(size[0], size[1])

def add_noise(image, noise):
    """Adds noise to image
    Parameters
    ----------
    image : np.array
        The input image
    noise : np.array
        The noise matrix
    Returns
    -------
    noised_image : np.array
        The noised_image
    """
    normalized_image = image/255.0
    return np.uint8((normalized_image + noise)/2.0 * 255.0)

if __name__=="__main__":
    # Loading Image
    img = load_image()
    # Problem 1
    swapped_img = switch_channels(img)
    save_image(swapped_img, "swapImgPS0Q2.png")
    # Problem 2
    gray_img = rgb2gray(img)
    save_image(gray_img, "grayImgPS0Q2.png")
    # Problem 3
    # Subproblem (a)
    neg_img = negative_image(gray_img)
    save_image(neg_img, "negativeImgPS0Q2.png")
    # Subproblem (b)
    flip_img = mirror_image(gray_img)
    save_image(flip_img, "mirrorImgPS0Q2.png")
    # Subproblem (c)
    avg_img = avg_image(gray_img, flip_img)
    save_image(avg_img, "avgImgPS0Q2.png")
    # Subproblem (d)
    noise = generate_noise(gray_img.shape)
    noised_img = add_noise(gray_img, noise)
    save_image(noised_img, "addNoiseImgPS0Q2.png")
    # Testing
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
    fig, axs = plt.subplots(3,2)
    # Subplot 1
    axs[0,0].imshow(swapped_img)
    axs[0,0].set_title(r'Image with Swapped Channels')
    # Subplot 2
    axs[0,1].imshow(gray_img, cmap="gray")
    axs[0,1].set_title(r'Gray Scale Image')
    # # Subplot 3
    axs[1,0].imshow(neg_img, cmap="gray")
    axs[1,0].set_title(r'Negative Image') 
    # # Subplot 4
    axs[1,1].imshow(flip_img, cmap="gray")
    axs[1,1].set_title(r'Mirror Image') 
    # Subplot 5
    axs[2,0].imshow(avg_img, cmap="gray")
    axs[2,0].set_title(r'Image averaged with its mirror') 
    # Subplot 6
    axs[2,1].imshow(noised_img, cmap="gray")
    axs[2,1].set_title(r'Noised Image') 
    
    # Labels
    fig.suptitle(r'\textbf{Image Transformations}')
    # Display plot
    plt.show()
