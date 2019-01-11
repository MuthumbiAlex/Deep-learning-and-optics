# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]



# MARCH 2018: code to read in my spherical simulation data that i created in matlab

def read_data_sets_microspheres(train_dir, 
                            fake_data=False, 
                            one_hot=False, 
                            dtype=dtypes.float32,
                            reshape=True,
                            validation_size=500): 
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.DataSet(train=train, validation=validation, test=test)

  #important parameter
  num_fp_images = 25

  # make a root folder where i will store the test and training data
  root_folder = '/Users/alex/Documents/WORK/germany/deep learning code/microspheres simulations/classification/full set/'

  ##############################################################################
  ### LOAD TEST AND TRAINING IMAGES

  #### TRAINING DATA
  # Step 1: get the images
  train_images = numpy.genfromtxt(root_folder + 'train_image_FP25_10k.txt')

  # Step 2: convert to uint8 for to reduce the size of the file
  train_images = numpy.uint8(train_images) # size is (10000, 19600)
  print('shape of the real part of FP train images')
  print(train_images.shape)

  # Step 3: reshape the real and imaginary parts
  svar = train_images.shape
  imdim_squared = svar[1]/num_fp_images # we are dividing 19600 by 25 = 784
  print(imdim_squared) # 784 as seen from above
  # Reshape the above training images into a 4D matrix, 1st dim as number of images, 2nd dim as size of images,
  # 3rd dim as the multiplication of sie of images and total number of LED : (10,000, 28, 700, 1)
  train_images = train_images.reshape(svar[0], numpy.int_(numpy.sqrt(imdim_squared)), num_fp_images * numpy.int_(numpy.sqrt(imdim_squared)), 1)
  print('training images reshaped into 2D array, with extra images concatenated along 3:')
  print(train_images.shape)


  ################# TEST IMAGES: repeat the same above for test images
  # Step 1: get the images
  test_images = numpy.genfromtxt(root_folder + 'test_image_FP25_2k.txt') # size is (2000, 19600)

  # Step 2: convert to uint8 to reduce size of files
  test_images = numpy.uint8(test_images) 
  print('shape of the test images')
  print(test_images.shape) # size is (2000, 19600)

  # Step 3: reshape the test images
  svar = test_images.shape # shape of test images is shown above
  imdim_squared = svar[1]/num_fp_images # perform 19600/25 = 784
  print(imdim_squared) # should be 784 as shown above

  # Reshape the above training images into a 4D matrix, 1st dim as number of images, 2nd dim as size of images,
  # 3rd dim as the multiplication of sie of images and total number of LED : (2000, 28, 700, 1)
  test_images = test_images.reshape(svar[0], numpy.int_(numpy.sqrt(imdim_squared)), num_fp_images * numpy.int_(numpy.sqrt(imdim_squared)), 1)
  print('test images reshapes into 2D array, with extra images concatenated along 3:')
  print(test_images.shape) 


  ################################################################
  ## LOAD TEST AND TRAINING LABELS
  
  train_labels = numpy.genfromtxt(root_folder + 'final_train_labels_changed.txt')
  train_labels = numpy.uint8(train_labels)
  test_labels = numpy.genfromtxt(root_folder + 'final_test_labels_changed.txt')
  test_labels = numpy.uint8(test_labels)

  print('shape of training labels') # 10,000 
  print(train_labels.shape)
  print('shape of test images') # 2000
  print(test_labels.shape)


  # Define number of classes 
  num_classes = 5
  if one_hot:
    test_labels = dense_to_one_hot(test_labels, num_classes)
    train_labels = dense_to_one_hot(train_labels, num_classes)

  validation_images = train_images[:validation_size] # means validation is running from 0 to val size i specified above, including the validation value
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:] # means training images run from val size + 1 till end
  train_labels = train_labels[validation_size:]

  # create a data set now
  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images, validation_labels, dtype = dtype, reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test = test)



def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)












