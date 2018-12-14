import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from matplotlib import pyplot as plt

print ' ===== Ship Detection In Satellite Practice ===== '

f = open('data/shipsnet.json')
dataset = json.load(f)
input_data = np.array(dataset['data']).astype('uint8')
output_data = np.array(dataset['labels']).astype('uint8')

print "Image Input Shape :: ", input_data.shape

# color chanel (RGB)
n_spectrum = 3
weight = 80
height = 80
X = input_data.reshape([-1, n_spectrum, weight, height])

# Each Channel Spectrum Disply
'''
pic = X[0]

rad_spectrum = pic[0]
green_spectrum = pic[1]
blue_spectum = pic[2]
plt.figure(1, figsize=(5 * 3, 5 * 1))
plt.set_cmap('jet')

show each channel
plt.subplot(1, 3, 1)
plt.imshow(rad_spectrum)

plt.subplot(1, 3, 2)
plt.imshow(green_spectrum)

plt.subplot(1, 3, 3)
plt.imshow(blue_spectum)
plt.show()
'''

# output encoding
y = np_utils.to_categorical(output_data, 2)
# shuffle all indexes
indexes = np.arange(4000)
np.random.shuffle(indexes)
X_train = X[indexes].transpose([0, 2, 3, 1])
y_train = y[indexes]
# normalization
X_train = X_train / 255

np.random.seed(100)

# network design
print "Network Design..."
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 40x40
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 20x20
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 10x10
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 5x5
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# optimization setup
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy'])

model_json = model.to_json()
with open('model/model.json', 'w') as json_file:
    json_file.write(model_json)
print "Saved Network model to disk"

print "Training Start..."
# training
model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=1000,
    validation_split=0.2,
    shuffle=True,
    verbose=2)
print "Training END..."

model.save_weights("model/model.h5")
print "Saved model weight to disk"