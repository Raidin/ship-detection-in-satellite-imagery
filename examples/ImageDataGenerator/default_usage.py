import numpy as np

from keras.preprocessing.image import ImageDataGenerator

np.random.seed(5)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.5,
    zoom_range=[0.8, 2.0],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'train_ship',
    target_size=(80, 80),
    batch_size=1,
    class_mode='categorical')
