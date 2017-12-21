import logging
import numpy as np
import os
import cv2
import h5py
import tensorflow as tf
from keras import backend as K


def open_image(fn, img_sz=None):
    if img_sz is None:
        return cv2.imread(fn)
    else:
        return cv2.resize(cv2.imread(fn), img_sz)


def save_hdf5(data, path):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset("data",  data=data)

def load_hdf5(path):
    with h5py.File(path, 'r') as hf:
        return hf['data'][:]


def limit_mem(gpu):
    gpu = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)
    logging.info('using GPU #{}'.format(gpu))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)
