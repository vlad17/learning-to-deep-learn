import keras
from keras import backend as K
from keras.layers import Reshape
import numpy as np
import tensorflow as tf

def refresh(seed=1234):
    """Kills the current keras session, restarts seeds. Assumes TF backend."""
    if K.backend() != 'tensorflow':
        raise RuntimeError('Expected tensorflow backend, found ' + K.backend())
    
    K.clear_session()
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
                
