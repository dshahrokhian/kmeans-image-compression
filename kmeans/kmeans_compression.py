# -*- coding: utf-8 -*-
"""
==========================================
K-Means Image Loss Compression - Algorithm
==========================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import numpy as np
from scipy import misc
from sklearn.cluster import KMeans


def reconstruct_image(codebook, labels, w, h):
    """ 
    Reconstructs the flat (1-dimensional) image into an image of dimensions (w,h).
    
    Parameters
    ----------
    codebook : Array
        Represents the different colors available for the image.
    labels : Array
        With len=w*h, it represents the position in codebook (this is, the color of 
        codebook) assigned to each pixel.
    w : int
        Width of the original image.
    h : int
        Height of the original image.
        
    Returns
    -------
    Array
        Reconstructed image.
    """
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[i*h+j]]
    
    return image
    
def compress(path, n_colors):
    """ 
    Compresses the color-palette of an image.
    
    Parameters
    ----------
    path : str
        Path to the image.
    n_colors : int
        Number of colors wanted in the compressed image.
        
    Returns
    -------
    Array
        Compressed image.
    """
    
    img = misc.imread(path)
    w, h, d = tuple(img.shape)
    assert d == 3, "Expected [R,G,B] pixel format"
    
    flat_img_array = np.reshape(img, (w * h, d)) # Reshape to 2D array for KMeans clustering
    
    model = KMeans(n_clusters=n_colors, random_state=0).fit(flat_img_array)
    flat_labels = model.predict(flat_img_array)
    
    return reconstruct_image(model.cluster_centers_, flat_labels, w, h)
