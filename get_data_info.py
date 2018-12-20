'''
Create At :: 2018.12.19
Create By :: JihunJung
Details
    Ship of Satellite Iamgery json data parsing and Infomation Display
'''
import json
import numpy as np
import pandas as pd

from keras.utils import np_utils
from matplotlib import pyplot as plt

def AnalysisData(dataset):
    print ' ===== AnalysisData ====='
    ships_net = pd.DataFrame(dataset)
    print 'Display the data in the first five rows...'
    print ships_net.head()
    print 'Display the data in the last thress rows...'
    print ships_net.tail(3)
    print 'ships_net index :: ', ships_net.index
    print 'ships_net columns :: ', ships_net.columns
    # print ships_net['locations']


def DescribeData(a, b):
    print ' ===== DescribeData ====='
    print ' - Total number of images: {}'.format(len(a))
    print ' - Number of NoShip Images: {}'.format(np.sum(b == 0))
    print ' - Number of Ship Images: {}'.format(np.sum(b == 1))
    print ' - Percentage of positive images: {:.2f}%'.format(100 * np.mean(b))
    print ' - Image shape (Width, Height, Channels): {}'.format(a[0].shape)

def DescribeDataset(features, labels):
    print ' ===== DescribeDataset ====='
    print " - 'Image' shape: %s." % (features.shape,)
    print " - 'Label' shape: %s." % (labels.shape,)
    print " - Unique elements in y: %s" % (np.unique(labels))

def plotOne(no_ship_img, ship_img, no_ship_idx=0, ship_idx=0):
    """
    Plot one numpy array
    """
    fig, axs = plt.subplots(1, 2, constrained_layout=True)

    axs[0].imshow(no_ship_img[no_ship_idx])
    axs[0].set_title('Not A Ship')
    axs[0].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])

    axs[1].imshow(ship_img[ship_idx])
    axs[1].set_title('Ship')
    axs[1].xaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])

    fig.canvas.set_window_title('Training Data Samples')
    fig.suptitle('Training Data Samples', fontsize=16)

def DipslayEachChannelSpectrum(images):
    # Each Channel Spectrum Disply
    pic = images[0]

    rad_spectrum = pic[0]
    green_spectrum = pic[1]
    blue_spectum = pic[2]
    plt.figure(1, figsize=(5 * 3, 5 * 1))
    plt.set_cmap('jet')

    # show each channel
    plt.subplot(1, 3, 1)
    plt.imshow(rad_spectrum)

    plt.subplot(1, 3, 2)
    plt.imshow(green_spectrum)

    plt.subplot(1, 3, 3)
    plt.imshow(blue_spectum)

def PlotHistogram(a, label):
    """
    Plot histogram of RGB Pixel Intensities
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(a)
    plt.axis('off')
    plt.title('Ship' if label else 'Not A Ship')
    histo = plt.subplot(1, 2, 2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:, :, 0].flatten(), bins=n_bins, lw=0, color='r', alpha=0.5)
    plt.hist(a[:, :, 1].flatten(), bins=n_bins, lw=0, color='g', alpha=0.5)
    plt.hist(a[:, :, 2].flatten(), bins=n_bins, lw=0, color='b', alpha=0.5)

def main():
    print ' ===== Json File Open ===== '
    f = open('data/shipsnet.json')
    dataset = json.load(f)
    # AnalysisData(dataset)

    # Convert to nparray type
    input_img = np.array(dataset['data']).astype('uint8')
    input_label = np.array(dataset['labels']).astype('uint8')
    DescribeData(input_img, input_label)

    # Reshape image and label
    # image : (4000, 19200) --> (4000, 3, 80, 80)
    # label : (4000,) --> (4000, 2)
    img_reshape = input_img.reshape([-1, 3, 80, 80]).transpose([0, 2, 3, 1])
    label_reshape = np_utils.to_categorical(input_label, num_classes=2)
    print " - Data Shape", input_img.shape
    print ' - Labels Shape', input_label.shape
    print ' - Reshaped Data Shape', img_reshape.shape
    print ' - Reshaped Labels Shape', label_reshape.shape
    DescribeDataset(img_reshape, label_reshape)

    # Store ship and no ship image
    no_ship_imgs = img_reshape[input_label == 0]
    ship_imgs = img_reshape[input_label == 1]

    plotOne(no_ship_imgs, ship_imgs)
    PlotHistogram(ship_imgs[100], 1)

    plt.show()


if __name__ == '__main__':
    main()
