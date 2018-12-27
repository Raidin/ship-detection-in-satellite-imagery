import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras import models
from keras import backend as K
from keras.models import model_from_json

def VisualizeFeaturemap(layer_name, layer_activation):
    layer_activation = np.squeeze(layer_activation, axis=0)
    layer_activation = layer_activation.transpose(2, 0, 1)

    fig, ax_arr = plt.subplots(3, 3, constrained_layout=True)
    fig.suptitle(layer_name, fontsize=16)
    for i in range(3):
        for j in range(3):
            ax_arr[i, j].imshow(layer_activation[(i * 3) + j], cmap='jet')
            ax_arr[i, j].xaxis.set_ticks([])
            ax_arr[i, j].yaxis.set_ticks([])
    fig.savefig('result/feature_map/{}.png'.format(layer_name))

def main():
    image = cv2.imread('/home/jihunjung/ship_detection/data/shipsnet/0__20160622_170157_0c64__-122.4478291812287_37.892486779170426.png')
    plt.imshow(image)
    plt.show()
    img_tensor = np.expand_dims(image, axis=0)

    # Load Network Model
    network_arch = 'defaultNet'
    json_file = open("model/{}/network_model.json".format(network_arch), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load Trained Weight
    model.load_weights("model/{}/trained_weight.h5".format(network_arch))
    print("Loaded model from disk")

    layer_outputs = [layer.output for layer in model.layers[:16]]
    # Creates a model that will return these outputs, given the model input
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    # Returns a list of five Numpy arrays: one array per layer activation
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for idx, layer in enumerate(model.layers[:12]):
        # Names of the layers, so you can have them as part of your plot
        layer_names.append('{}_{}'.format(idx, layer.name))

    # Displays the feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        VisualizeFeaturemap(layer_name, layer_activation)

    print 'Intermediate Feature Map Display Completed...'
if __name__ == '__main__':
    main()
