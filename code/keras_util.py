import keras
from keras import backend as K
from keras.layers import Reshape
import keras.backend.tensorflow_backend
import numpy as np
import tensorflow as tf
import ray

from contextlib import closing, contextmanager, closing
from functools import lru_cache
from multiprocessing import get_context
import os
import pickle
import time
import subprocess
import sys

if K.backend() != 'tensorflow':
    raise RuntimeError('Expected tensorflow backend, found ' + K.backend())


def seedall(seed=1234):
    np.random.seed(seed)
    tf.set_random_seed(seed)


class EarlyStopLambda(keras.callbacks.Callback):
    def __init__(self, metric='loss', should_stop=None):
        super(EarlyStopLambda, self).__init__()
        self.metric = metric
        self.should_stop = should_stop

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.metric)
        if self.should_stop(current):
            self.model.stop_training = True


@contextmanager
def rectime(fmt='{:4.0f} sec'):
    t = time.time()
    yield
    t = time.time() - t
    print(fmt.format(t))


def keras_tf_active():
    return keras.backend.tensorflow_backend._SESSION is not None


@lru_cache(maxsize=1)
def ngpu():
    # Do this in a subprocess to avoid creating a session here.
    cmd = 'from tensorflow.python.client import device_lib;'
    cmd += 'local_device_protos = device_lib.list_local_devices();'
    cmd += """print(sum(device.device_type == 'GPU' for device in local_device_protos))"""
    out = subprocess.check_output(
        [sys.executable, '-c',  cmd], stderr=subprocess.DEVNULL)
    return int(out.strip())


@contextmanager
def cd(new_dir):
    prev_dir = os.getcwd()
    os.makedirs(new_dir, exist_ok=True)
    os.chdir(new_dir)
    yield
    os.chdir(prev_dir)


def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


@contextmanager
def ray_gpu_session():
    gpus = ",".join(map(str, ray.get_gpu_ids()))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    with tf.Session() as sess:
        with sess.as_default():
            K.set_session(sess)
            yield
