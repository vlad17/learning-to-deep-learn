# File modified by Vladimir Feinberg
# =============================================================================
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Functions for downloading and reading MNIST data.

File originated from TensorFlow repository:
https://github.com/tensorflow/tensorflow

Relative path to file from root, at commit
41e363eda80d31c1ce67f5f646a11c185eba9635:
tensorflow/examples/tutorials/mnist/input_data.py
"""

import gzip
import os

import numpy as np
from six.moves import urllib

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def _download_if_absent(filename, work_directory):
    """Download the data from Yann's website, unless it's already in the work
    directory. Creates the directory if it does not exist. Prints annoying
    diagnostic info, but just the right amount"""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(
            SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_images(filename):
    """Extract the images into a 4D uint8 np array [index, y, x, depth].
    Note that for MNIST depth == 1."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def _dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros(
        (num_labels, num_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def _extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 np array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return _dense_to_one_hot(labels)
        return labels


def _rescale_and_retype(images, exact):
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    assert images.shape[3] == 1
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2])
    if not exact:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    return images


def read_data_sets(train_dir, one_hot=False, exact_inputs=False):
    """
    If necessary, downloads MNIST data into train_dir. Using the downloaded
    files in the given directory, extracts the MNIST data and returns a tuple
    (training images, training labels, testing images, testing labels),
    where image arrays are tensors with size (n_examples, width * height).

    If one_hot is true, then testing arrays have size (n_examples, 10),
    else they are 1D arrays with the integral class value between 0 and 9.

    If exact_inputs is set to true, images are kept in uint8 data type form
    for their grey scale values [0, 256). Otherwise, they are 4-byte floats
    scaled to 0 and 1.
    """

    train_images_filename = 'train-images-idx3-ubyte.gz'
    train_labels_filename = 'train-labels-idx1-ubyte.gz'
    test_images_filename = 't10k-images-idx3-ubyte.gz'
    test_labels_filename = 't10k-labels-idx1-ubyte.gz'

    local_file = _download_if_absent(train_images_filename, train_dir)
    train_images = _extract_images(local_file)
    local_file = _download_if_absent(train_labels_filename, train_dir)
    train_labels = _extract_labels(local_file, one_hot=one_hot)
    local_file = _download_if_absent(test_images_filename, train_dir)
    test_images = _extract_images(local_file)
    local_file = _download_if_absent(test_labels_filename, train_dir)
    test_labels = _extract_labels(local_file, one_hot=one_hot)

    train_images = _rescale_and_retype(train_images, exact_inputs)
    test_images = _rescale_and_retype(test_images, exact_inputs)

    return (train_images, train_labels, test_images, test_labels)
