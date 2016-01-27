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
import tensorflow.python.platform
import numpy
from itertools import *
from six.moves import urllib
import tensorflow as tf
from data_set import DataSet

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def download_if_absent(filename, work_directory):
  """Download the data from Yann's website, unless it's already in the work
  directory. Creates the directory if it does not exist. Prints annoying
  diagnostic info, but just the right amount"""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
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
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes), dtype=numpy.float32)
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels

def rescale_and_retype(images, exact):
  # Convert shape from [num examples, rows, columns, depth]
  # to [num examples, rows*columns] (assuming depth == 1)
  assert images.shape[3] == 1
  images = images.reshape(images.shape[0],
                          images.shape[1] * images.shape[2])
  if not exact:
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(numpy.float32)
    images = numpy.multiply(images, 1.0 / 255.0)

  return images

def read_data_sets(train_dir, one_hot=False, exact_inputs=False):
  """If necessary, downloads MNIST data into train_dir. Using the downloaded
  files in the given directory, extracts the MNIST data and returns a tuple of
  (training images, training labels, testing images, testing labels),
  where image arrays are 4D tensors and labels are 1D if dense encoding and
  2D otherwise (one-hot encoding).

  If exact_inputs is set to true, images are kept in uint8 data type form
  for their pixel values [0, 256). Otherwise, they are 4-byte floats
  scaled to 0 and 1.
"""

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = download_if_absent(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = download_if_absent(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)
  local_file = download_if_absent(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = download_if_absent(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  train_images = rescale_and_retype(train_images, exact_inputs)
  test_images = rescale_and_retype(test_images, exact_inputs)

  return (DataSet(train_images, train_labels),
          DataSet(test_images, test_labels))

          
