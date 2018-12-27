"""
Create at : 2018. 12. 14
Writer : jihunjung

usage: train.py [-h] [--val-split VAL_SPLIT] [--batch-size BATCH_SIZE]
                [--epochs EPOCHS] [--shuffle SHUFFLE]
                [--learning-rate LEARNING_RATE]
                job_name

Process some parameters.

positional arguments:
  job_name              Current operation job name

optional arguments:
  -h, --help            show this help message and exit
  --val-split VAL_SPLIT
                        Training Parameter-Validation set
                        percentage(Type:float, default Value:0.2)
  --batch-size BATCH_SIZE
                        Training Parameter-batch size(Type:integer, default
                        Value:32)
  --epochs EPOCHS       Training Parameter-Epoch(Type:integer, default
                        Value:20)
  --shuffle SHUFFLE     Training Parameter-Is Apply Suffle(Type:bool, default
                        Value:True)
  --learning-rate LEARNING_RATE
                        Training Parameter-Learning Rate(Type:float, default
                        Value:1e-2)

"""
import argparse
import generation_model
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from common import make_if_not_exist
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

    color_list = ['r', 'g', 'b', 'y']

    for idx, hist_key in enumerate(hist.history.keys()):
        if 'loss' in hist_key:
            loss_ax.plot(hist.history[hist_key], color_list[idx], label=hist_key)
        elif 'acc' in hist_key:
            acc_ax.plot(hist.history[hist_key], color_list[idx], label=hist_key)

    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    acc_ax.set_ylabel('Accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    fig.suptitle('Training Results(Accuracy & Loss)', fontsize=16)
    fig.savefig('{}/training_result_plot.png'.format(save_dir))

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

def main(config):
    # 1.Load Dataset
    print '1. Load Dataset...'
    image_train, label_train = LoadDataset(config['root-dir'])

    # define numpy seed
    np.random.seed(42)

    # 2.Generate Model
    print '2. Generate Model...'
    model = eval('generation_model.{}()'.format(config['job-name']))
    # optimization setup
    sgd = SGD(lr=config['learning-rate'], momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # 3.save network model
    print '3. Save Network Model...'
    SaveNetworkModel(model, config['model-dir'])
    csv_logger = CSVLogger('{}/log.csv'.format(config['model-dir']), append=True, separator=';')

    # 4.training
    print '4. Training Network Model...'
    history = model.fit(image_train, label_train, batch_size=config['batch-size'], epochs=config['epochs'],
                            validation_split=config['val-split'], shuffle=config['shuffle'],
                            verbose=2, callbacks=[csv_logger])

    # 5.save trained weight
    print '5. Save Trained Weight(*.h5)...'
    SaveWeight(model, config['model-dir'])
    # 6.save training result(accuracy/loss) plot
    print '6. Save Acc/Loss Plot...'
    VisualizationPlot(history, config['model-dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('job_name', help='Current operation job name')
    parser.add_argument('--val-split', type=float, default=0.2, help='Training Parameter-Validation set percentage(Type:float, default Value:0.2)')
    parser.add_argument('--batch-size', type=int, default=32, help='Training Parameter-batch size(Type:integer, default Value:32)')
    parser.add_argument('--epochs', type=int, default=20, help='Training Parameter-Epoch(Type:integer, default Value:20)')
    parser.add_argument('--shuffle', type=bool, default=True, help='Training Parameter-Is Apply Suffle(Type:bool, default Value:True)')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Training Parameter-Learning Rate(Type:float, default Value:1e-2)')
    args = parser.parse_args()

    cur_dir = os.getcwd()
    root_dir = os.path.abspath(os.path.join(cur_dir, ".."))
    work_dir = os.path.join(root_dir, 'work')
    jobs_dir = os.path.join(work_dir, args.job_name)
    model_dir = os.path.join(jobs_dir, 'model')

    config = {'job-name': args.job_name,
                'val-split': args.val_split,
                'batch-size': args.batch_size,
                'epochs': args.epochs,
                'shuffle': args.shuffle,
                'learning-rate': args.learning_rate,
                'root-dir': root_dir,
                'work-dir': work_dir,
                'jobs-dir': jobs_dir,
                'model-dir': model_dir}

    print '\n\n::::: Configuration Value :::::'
    for config_key in config.keys():
        print ' - {} :: {}'.format(config_key, config[config_key])

    make_if_not_exist(work_dir)
    make_if_not_exist(jobs_dir)
    make_if_not_exist(model_dir)

    main(config)
