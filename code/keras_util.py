import keras
from keras import backend as K
from keras.layers import Reshape
import keras.backend.tensorflow_backend
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from contextlib import closing, contextmanager, closing
from multiprocessing import get_context
import os
import pickle
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
def rectime(name='', fmt='{: 4.0f}'):
    print(name, end='')
    sys.stdout.flush()
    t = time.time()
    yield
    t = time.time() - t
    print(fmt.format(t), 'sec')


def keras_tf_active():
    return keras.backend.tensorflow_backend._SESSION is not None


def get_available_gpus():
    # https://gist.github.com/jovianlin/b5b2c4be45437992df58a7dd13dbafa7
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


@contextmanager
def tmp_cd(new_dir):
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

# --- old, broken code below


def gpu_map(f, args, pass_update_callback=False):
    args = [(arg,) for arg in args]
    return gpu_starmap(f, args, pass_update_callback)


def gpu_starmap(f, args, pass_update_callback=False):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True))
    server = tf.train.Server.create_local_server(config=config)
    ctx = get_context('spawn')  # need to spawn b/c of TF global state
    manager = ctx.Manager()
    devices = get_available_gpus()
    ngpu = len(devices)
    device_queue = manager.Queue(maxsize=ngpu)
    for device in devices:
        device_queue.put_nowait(device)

    pickle_f = pickle.dumps(f)
    callback_queue = manager.Queue()
    args = [(callback_queue, pass_update_callback, server.target, device_queue, pickle_f, arg_tup)
            for arg_tup in args + [None]]

    K.clear_session()
    with closing(ctx.Pool(processes=ngpu)) as pool:
        result = pool.starmap_async(_gpu_starmap_work, args, chunksize=1)
        while True:
            msg = callback_queue.get()
            if msg is None:  # sentinel returned
                return result.get()
            print(msg)


def _gpu_starmap_work(callback_queue, pass_update_callback, server_target, device_queue, pickle_f, arg_tup):
    if arg_tup is None:  # Hit sentinel
        callback_queue.put(None)
    with _queue_borrow(device_queue) as device:
        old_arg_tup = arg_tup

        def callback(msg):
            callback_queue.put(
                '{} task {}: {}'.format(device, old_arg_tup, msg))
        if pass_update_callback:
            arg_tup = (callback,) + old_arg_tup
        callback('made callback')
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=device[len('/gpu:'):]))
        callback('made config')
        callback('made gpuopts {}'.format(config))
        with tf.Session(server_target, config=config, graph=tf.Graph()) as sess:
            with sess.as_default():
                callback('made sess')
                import keras.backend as K
                K.set_session(sess)
                callback('made keras')
                import keras
                f = pickle.loads(pickle_f)
                callback('made calling f')
                return f(*arg_tup)


@contextmanager
def _queue_borrow(queue):
    borrowed = queue.get()
    yield borrowed
    queue.put_nowait(borrowed)
