from io import StringIO
from itertools import chain
import sys

from keras import backend as K
from keras.layers.core import Lambda
from keras.models import Model
from keras.layers.merge import Concatenate
import keras.optimizers

if K.backend() != 'tensorflow':
    raise RuntimeError('Expected tensorflow backend, found ' + K.backend())

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.ops import data_flow_ops

try:
    from tensorflow.contrib import nccl
    have_nccl = True
    print('NCCL support available', file=sys.stderr)
except ImportError:
    have_nccl = False
    print('WARNING: NCCL support not available', file=sys.stderr)


def _get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpus_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return gpus_list


def _all_sync_params(tower_params, devices, usenccl=True):
    """Assigns the params from the first tower to all others"""
    if len(devices) == 1:
        return tf.no_op()
    sync_ops = []
    if have_nccl and usenccl:
        for param_on_devices in zip(*tower_params):
            param0 = param_on_devices[0]
            send_op, received_tensors = nccl.broadcast(param0, devices[1:])
            sync_ops.append(send_op)
            for device, param, received in zip(devices[1:],
                                               param_on_devices[1:],
                                               received_tensors):
                with tf.device(device):
                    sync_op = param.assign(received)
                    sync_ops.append(sync_op)
    else:
        params0 = tower_params[0]
        for device, params in zip(devices, tower_params):
            with tf.device(device):
                for param, param0 in zip(params, params0):
                    sync_op = param.assign(param0.read_value())
                    sync_ops.append(sync_op)

    return tf.group(*sync_ops)


class SyncMGPU(Model):

    def __init__(self, *args, **kwargs):
        try:
            smodel = kwargs.pop('serial_model')
        except KeyError:
            raise RuntimeError('Keyword argument "serial_model" required '
                               'for SyncMGPU.')

        # SET STATE: Instance of serial model for checkpointing
        self._smodel = smodel

        try:
            gdev_list = kwargs.pop('gdev_list')
        except KeyError:
            raise RuntimeError('Keyword argument "gdev_list" required '
                               'for SyncMGPU.')
        self._gdev_list = gdev_list

        mname = kwargs.pop('name', self._smodel.name)
        kwargs['name'] = mname

        self._ps_device = kwargs.pop('ps_device', '/cpu:0')
        self._usenccl = kwargs.pop('usenccl', False)

        self._tower_params = []  # For init/sync'ing of parameters.
        self._init_make_dataparallel(gdev_list, *args,
                                     **kwargs)

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(SyncMGPU, self).__getattribute__(attrname)

    # ref: https://github.com/fchollet/keras/issues/2436
    def _init_make_dataparallel(self, gdev_list, *args, **kwargs):
        '''Uses data-parallelism to convert a serial model to multi-gpu. Refer
        to make_parallel doc.
        '''
        def slice_batch(x, ngpus, part, dev):
            '''Divide the input batch into [ngpus] slices, and obtain slice
            no. [part]. i.e. if len(x)=10, then slice_batch(x, 2, 1) will
            return x[5:].
            '''
            sh = K.shape(x)
            L = sh[0] // ngpus
            if part == ngpus - 1:
                xslice = x[part * L:]
            else:
                xslice = x[part * L:(part + 1) * L]

            return xslice

        ngpus = len(gdev_list)
        if ngpus < 2:
            raise RuntimeError('Number of gpus < 2. Require two or more GPUs '
                               'for multi-gpu model parallelization.')

        model_ = model = self._smodel
        global_scope = tf.get_variable_scope()
        towers = []
        for idev, dev in enumerate(gdev_list):
            with tf.device(self._ps_device):
                slices = []  # multi-input case
                for ix, x in enumerate(model.inputs):
                    slice_g = Lambda(
                        slice_batch,
                        name='stage_cpuSliceIn{}_Dev{}'.format(ix, idev),
                        arguments={'ngpus': ngpus, 'part': idev,
                                   'dev': dev})(x)
                    slices.append(slice_g)
            with tf.device(dev), \
                    tf.variable_scope(global_scope, reuse=idev > 0), \
                    tf.name_scope('tower_%i' % idev):

                modeltower = model_(slices)
                towers.append(modeltower)

                params = modeltower.graph._collections['trainable_variables']

                self._tower_params.append(params)

        with tf.device(self._ps_device):
            merged = Concatenate(axis=0)(towers)
            # print('MERGED: {}'.format(merged))  # DEBUG

        kwargs['inputs'] = model.inputs
        kwargs['outputs'] = merged
        super(SyncMGPU, self).__init__(*args, **kwargs)

    def compile(self, *args, **kwargs):
        '''Refer to Model.compile docstring for parameters. Override
        functionality is documented below.

        :override compile: Override Model.compile method to check for options
            that the optimizer is multi-gpu enabled, and synchronize initial
            variables.
        '''
        usenccl = self._usenccl

        opt = kwargs['optimizer']
        if isinstance(opt, str):
            opt = keras.optimizers.get(opt)
            kwargs['optimizer'] = opt

        opt.usenccl = usenccl

        super(SyncMGPU, self).compile(*args, **kwargs)

        self._run_initsync()

    def _run_initsync(self):
        tparams = self._tower_params

        # Check to prevent from unnecessarily re-initializing and
        # synchronizing, i.e. when the model loads the weights.
        for v in chain.from_iterable(tparams):
            if getattr(v, '_keras_initialized', False):
                return

        K.manual_variable_initialization(True)
        sess = K.get_session()
        K.manual_variable_initialization(False)

        # Initialize on GPU0 and sync to other GPUs
        init_op = tf.variables_initializer(tparams[0])
        sess.run(init_op)
        sync_op = _all_sync_params(tparams, self._gdev_list,
                                   usenccl=self._usenccl)
        sess.run(sync_op)

        for v in chain.from_iterable(tparams):
            v._keras_initialized = True


def make_parallel(serial_model, gdev_list=None, ps_device='/cpu:0',
                  usenccl=True, model_class=SyncMGPU):

    if gdev_list is None:
        gdev_list = _get_available_gpus()

    ngpus = len(gdev_list)
    if ngpus < 2:
        return serial_model

    return model_class(
        serial_model=serial_model, gdev_list=gdev_list,
        ps_device=ps_device,
        usenccl=usenccl)
