import json
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.preprocessing import minmax_scale # [0-1] Scaling


def LoadDataset(root):
    data = os.path.join(root, 'data/shipsnet.json')
    f = open(data)
    dataset = json.load(f)

    input_data = np.array(dataset['data']).astype('float64')
    output_data = np.array(dataset['labels']).astype('uint8')

    # color chanel (RGB)
    channel = 3
    weight = 80
    height = 80

    # normalization to [0~1] (Using min/max scaling)
    input_data = minmax_scale(input_data, feature_range=(0, 1), axis=0)

    # input image & label reshape
    images = input_data.reshape([-1, channel, weight, height]).transpose([0, 2, 3, 1])

    labels = np_utils.to_categorical(output_data, 2)

    # shuffle all indexes
    indexes = np.arange(4000)
    np.random.shuffle(indexes)
    image_train = images[indexes]
    label_train = labels[indexes]

    return image_train, label_train


cur_dir = os.getcwd()
project_root_dir = os.path.abspath(os.path.join(cur_dir, "../.."))
image_train, label_train = LoadDataset(project_root_dir)

print image_train.shape
print label_train.shape

np.random.seed(3)

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            rotation_range=90,
                                            shear_range=0.5,
                                            zoom_range=[0.8, 1.5],
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='nearest',
                                            validation_split=0.2)

train_datagen.fit(image_train)

train_generator = train_datagen.flow(image_train, label_train,
                                            batch_size=32, shuffle=True,
                                            subset='training')

validation_generator = train_datagen.flow(image_train, label_train,
                                            batch_size=32, shuffle=True,
                                            subset='validation')


# Construct Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(80, 80, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

#Set Training Process
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fits the model on batches with real-time data augmentation:
'''
* Total Train Image : 4000(3000/1000)
* Divide
  - Train : 3200(2400, 800)
  - Val : 800(600, 200)

* Train Batch : 32
  - Train Step per Epoch : (3200/32) * 5
  - Val Step per Epoch : (800/32)
'''

model.fit_generator(
    train_generator,
    steps_per_epoch=100 * 5,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=25,
    verbose=2)
