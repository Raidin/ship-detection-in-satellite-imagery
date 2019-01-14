import os
import sys
import numpy as np
from PIL import Image

image = Image.open('../../data/scenes/sfbay_2.png')
pix = image.load()

n_spectrum = 3
width = image.size[0]
height = image.size[1]
# creat vector
picture_vector = []
for channel in range(n_spectrum):
    for y in range(height):
        for x in range(width):
            picture_vector.append(pix[x, y][channel])

picture_vector = np.array(picture_vector).astype('uint8')
picture_tensor = picture_vector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)

# Convert np array to Image
re_image = Image.fromarray(picture_tensor)
re_image.save('input_image.png', 'PNG')
