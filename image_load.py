import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

image = Image.open('/home/jihunjung/Downloads/Satellite_Image_Dataset_with_ship_plane/scenes/sfbay_2.png')
pix = image.load()
n_spectrum = 3
width = image.size[0]
height = image.size[1]
# creat vector
picture_vector = []
for chanel in range(n_spectrum):
    for y in range(height):
        for x in range(width):
            picture_vector.append(pix[x, y][chanel])

picture_vector = np.array(picture_vector).astype('uint8')
picture_tensor = picture_vector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)

print picture_tensor.shape

plt.figure(1, figsize=(15, 30))
plt.imshow(picture_tensor)
plt.show()