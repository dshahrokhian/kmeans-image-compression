# -*- coding: utf-8 -*-
"""
=====================================
K-Means Image Loss Compression - Test
=====================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import numpy as np
import matplotlib.pyplot as plt
import kmeans.kmeans_compression as kmeans
from io import BytesIO
from PIL import Image

def calc_file_size(img):
    img_array = Image.fromarray(img,'RGB')
    output = BytesIO()
    img_array.save(output, 'PNG')
    size = output.tell()
    output.close()

    return size

def save(img):
    
    # Convert from 8-bit integers to floats for plt.imshow to work in the range of [0,1]
    img = np.array(img, dtype=np.float64) / 255
    
    plt.imshow(img)
    plt.annotate("Number of colors: " + str(n_colors) + 
                 "\nApprox file size: " + str(round(calc_file_size(img)/1024, 1)) + " kB",
                 xy=(.25, .25), 
                 xytext=(40, 100), 
                 fontsize=10, 
                 bbox={'facecolor':'white', 'alpha':0.85, 'pad':5})
    plt.axis('off')
    plt.savefig(str(n_colors) + "colors.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    image_path = "futur.png"
    
    n_colors = 64
    while n_colors > 8:
        compressed_image = kmeans.compress(image_path, n_colors)
        save(compressed_image)
        n_colors //= 2
    
    while n_colors > 1:
        compressed_image = kmeans.compress(image_path, n_colors)
        save(compressed_image)
        n_colors -= 1
