import numpy as np
import os
import sys

from tqdm import tqdm
from PIL import Image

sys.path.insert(1, '../')
from common import make_if_not_exist, Numpy2Image

import cv2

img = cv2.imread('/home/jihunjung/Downloads/satellite_output_2.jpg')

print img.shape

sys.exit()

def CropImage(img, x, y, window_size):
    area_study = np.arange(3 * window_size * window_size).reshape(window_size, window_size, 3).astype('uint8')
    for i in range(window_size):
        for j in range(window_size):
            area_study[i][j][0] = img[y + i][x + j][0]
            area_study[i][j][1] = img[y + i][x + j][1]
            area_study[i][j][2] = img[y + i][x + j][2]
    return area_study


cur_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(cur_dir, "../.."))
data_dir = os.path.join(root_dir, 'data/scenes')
out_dir = os.path.join(root_dir, 'data/scenes_crop')
make_if_not_exist(out_dir)

step = 256
window_size = 256

# Test Image list Load
for idx, img_file in tqdm(enumerate(os.listdir(data_dir))):
    input_image = os.path.join(data_dir, img_file)
    sub_dir = os.path.join(out_dir, os.path.splitext(img_file)[0])
    make_if_not_exist(sub_dir)

    image = Image.open(input_image)
    pix = image.load()
    n_spectrum = 3
    width = image.size[0]
    height = image.size[1]

    picture_vector = []
    for chanel in range(n_spectrum):
        for y in range(height):
            for x in range(width):
                picture_vector.append(pix[x, y][chanel])

    picture_vector = np.array(picture_vector).astype('uint8')
    # Shape :: channel x hegith x widht
    # picture_tensor = picture_vector.reshape([n_spectrum, height, width])
    picture_tensor = picture_vector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)

    cnt = 0
    for y in tqdm(range(int((height - (window_size - step)) / step))):
        for x in range(int((width - (window_size - step)) / step)):
            area = CropImage(picture_tensor, x * step, y * step, window_size)
            Numpy2Image(area, os.path.join(sub_dir, '{}.png'.format(cnt)))
            cnt = cnt + 1
