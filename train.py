import generation_model
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.utils import np_utils
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import CSVLogger

# Feature scaling import
from sklearn.preprocessing import minmax_scale # [0-1] Scaling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def SaveNetworkModel(model, save_dir):
    # Model to Save File
    # To Json
    model_json = model.to_json()
    with open('{}/network_model.json'.format(save_dir), 'w') as json_file:
        json_file.write(model_json)
    # To Text File
    with open('{}/network_model.txt'.format(save_dir), 'w') as model_file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: model_file.write(x + '\n'))

    # To Model Visualization
    plot_model(model, to_file='{}/model_plot.png'.format(save_dir), show_shapes=True, show_layer_names=True)
    print "Saved Network model to disk..."

def SaveWeight(model, save_dir):
    model.save_weights("{}/trained_weight.h5".format(save_dir))
    print "Saved training weight to disk..."

def VisualizationPlot(hist, save_dir):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    acc_ax.set_ylabel('Accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    fig.suptitle('Training Results(Accuracy & Loss)', fontsize=16)
    fig.savefig('{}/training_result_plot.png'.format(save_dir))

def LoadDataset():
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

def main():
    print ' ===== Ship Detection In Satellite Practice ===== '
    network_arch = 'defaultNet_2'
    save_dir = os.path.join('model', network_arch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load Dataset
    image_train, label_train = LoadDataset()

    np.random.seed(42)

    # network design
    if network_arch == 'defaultNet_2':
        model = generation_model.DefaultNet()
    elif network_arch == 'simpleNet_01':
        model = generation_model.SimpleNet_01()()

    # optimization setup
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # save network model
    SaveNetworkModel(model, save_dir)

    csv_logger = CSVLogger('{}/log.csv'.format(save_dir), append=True, separator=';')

    print "Training Start..."
    # training
    # history = model.fit(image_train, label_train, batch_size=32, epochs=30, validation_split=0.2, shuffle=True, verbose=2, callbacks=[csv_logger])
    history = model.fit(image_train, label_train, batch_size=32, epochs=200,
                            validation_split=0.2, shuffle=True, verbose=2, callbacks=[csv_logger])

    print "Training END..."

    # save trained weight
    SaveWeight(model, save_dir)
    # save training result(accuracy/loss) plot
    VisualizationPlot(history, save_dir)


if __name__ == '__main__':
    main()
