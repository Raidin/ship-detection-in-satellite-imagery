import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

ship = cv2.imread('/home/jihunjung/ship_detection/data/shipsnet/0__20160710_182140_0c78__-122.33743457924228_37.75728971371875.png')

plt.imshow(ship)

print ship.shape

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=ship.shape))
ship_batch = np.expand_dims(ship, axis=0)
print ship_batch.shape
conv_ship = model.predict(ship_batch)

def vis_featuremap(ship_batch):
    ship = np.squeeze(ship_batch, axis=0)
    print ship.shape
    ship = ship.transpose(2, 0, 1)
    print ship.shape

    fig, ax_arr = plt.subplots(3, 3, constrained_layout=True)
    for i in range(3):
        for j in range(3):
            ax_arr[i, j].imshow(ship[(i * 3) + j], cmap='jet')
            ax_arr[i, j].xaxis.set_ticks([])
            ax_arr[i, j].yaxis.set_ticks([])

print conv_ship.shape
vis_featuremap(conv_ship)
plt.show()
