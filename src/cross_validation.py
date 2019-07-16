import numpy as np
import generation_model
import json

from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale

def create_model():
    # create model
    model = generation_model.DefaultNet()

    # domplie model
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def dataset_load():
    print ' - Json File Open...'
    f = open('data/shipsnet.json')
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

# Fix random seed for reproductibility
seed = 42
np.random.seed(seed)

# Load Ship-Detection Dataset :: Ship 1000, No Ship : 3000 --> Total : 4000
X, Y = dataset_load()

# Create Model
model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=32, verbose=2)

# Set K-Fold
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)

# print results
for i in range(len(results)):
    print '{} :: {}'.format(i, results[i])
