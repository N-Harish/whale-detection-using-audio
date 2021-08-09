import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from librosa.feature import melspectrogram, mfcc
from tqdm import tqdm
import librosa
import warnings
import h5py
import numpy as np
import os


# Ignore all warnings
warnings.filterwarnings("ignore")


def mel_spec_gen(path_csv, labels, class_name, data_dir, output_dir):
    """

    :param output_dir: path to save output images
    :param data_dir: path of audio file
    :param class_name: class name to store generated image
    :param labels: To filter dataframe based on passed label
    :param path_csv: path of csv containing data
    :type path_csv: path
    :type labels: int
    :type class_name: str
    :type data_dir: path
    :type output_dir: path

    """
    df = pd.read_csv(path_csv)
    df1 = df[df['label'] == labels]
    label = df1['label']
    df1 = df1['clip_name'].apply(lambda x: x.replace('aiff', 'wav'))
    df1 = pd.DataFrame({'filename': df1.to_list(), 'label': label.to_list()})

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    li = os.listdir(output_dir)
    li = [str(i).replace('jpg', 'wav') for i in li]
    df1 = df1[~df1.filename.isin(li)]

    # Give spacing for progress bar by print statement
    print('================== \n')
    print(f'Generate mel spectogram for audio containing {class_name}:- \n')

    # pass total length in tqdm
    # pass colour in tqdm to display coloured progress bar
    for i in tqdm(df1.itertuples(), colour='green', total=df1.shape[0]):
        y, sr = librosa.load(f'{data_dir}/{i.filename}')
        plot = melspectrogram(y, sr, n_mels=13)
        plot = librosa.power_to_db(plot, ref=np.max)
        librosa.display.specshow(plot, y_axis='mel', x_axis='time')
        plt.tight_layout()
        fname = i.filename
        fname = str(fname).replace('wav', 'jpg')
        plt.savefig(f'{output_dir}/{fname}')

    print(f'====completed mel spectogram of {class_name}====\n')


def mfcc_gen(path_csv: str, data_dir: str, feature_path: str, label_path: str):
    """

    :type data_dir: path
    :type path_csv: path
    :type feature_path: path
    :type label_path: path
    :param label_path: path to save label file
    :param feature_path: path to save feature file
    :param data_dir: path of audio file
    :param path_csv: path of csv file containing data

    """
    df = pd.read_csv(path_csv)
    label = df['label']
    df = df['clip_name'].apply(lambda x: x.replace('aiff', 'wav'))
    df = pd.DataFrame({'filename': df.to_list(), 'label': label.to_list()})
    mfcc_fts = []
    labels = []

    print(f'\nGenerate mel frequency ceptral coefficient (mfcc) for audio:-')

    for i in tqdm(df.itertuples(), colour='green', total=df.shape[0]):
        y, sr = librosa.load(f'{data_dir}/{i.filename}')
        mfcc_ft = mfcc(y, sr).flatten()
        mfcc_fts.append(mfcc_ft)
        labels.append(i.label)

    # Create h5py file for features
    with h5py.File(feature_path, 'w') as h5_data:
        h5_data.create_dataset('dataset', data=np.array(mfcc_fts))

    # create h5py  file for labels
    with h5py.File(label_path, 'w') as h5_label:
        h5_label.create_dataset('dataset', data=np.array(labels))

    print(f'====completed mel frequency ceptral coefficient (mfcc) ====')


def test_feature(feature_file, data_dir):
    mfcc_fts = []

    for i in tqdm(os.listdir(data_dir), colour='green'):
        y, sr = librosa.load(f'{data_dir}/{i}')
        mfcc_ft = mfcc(y, sr).flatten()
        mfcc_fts.append(mfcc_ft)

    with h5py.File(feature_file, 'w') as h5_data:
        h5_data.create_dataset('dataset', data=np.array(mfcc_fts))
