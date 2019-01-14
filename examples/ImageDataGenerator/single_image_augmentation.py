import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

if not os.path.exists('augmentation_sample/'):
    os.makedirs('augmentation_sample/')

np.random.seed(3)

cur_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(cur_dir, ".."))

img = load_img('/home/jihunjung/ship_detection/data/shipsnet/1__20170910_181216_1010__-122.32449564687262_37.7270132374854.png')
img.save('augmentation_sample/origin.png', 'PNG')
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

i = 0

np.random.seed(5)

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            rotation_range=90,
                                            shear_range=0.5,
                                            zoom_range=[0.8, 1.5],
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='nearest')

for batch in train_datagen.flow(x, batch_size=1, save_to_dir='augmentation_sample/', save_prefix='tri', save_format='png'):
    i += 1
    if i > 4:
        break
