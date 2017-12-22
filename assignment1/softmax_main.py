import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import time

import sys
sys.path.insert(0, sys.path[0]+'\\classifier')
from softmax import softmax_loss_naive
from softmax import softmax_loss_vectorized
from linear_classifier import Softmax


#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_CIFAR10_data(num_training=1000, num_validation=100, num_test=500):
  """
  Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
  it for the linear classifier. These are the same steps as we used for the
  SVM, but condensed to a single function.  
  """
  # Load the raw CIFAR-10 data
  cifar10_dir = 'datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]
  
  # Preprocessing: reshape the image data into rows
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))
  
  # Normalize the data: subtract the mean image
  mean_image = np.mean(X_train, axis = 0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image
  
  # add bias dimension and transform into columns
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
  
  return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

##print 'Train data shape: ', X_train.shape
##print 'Train labels shape: ', y_train.shape
##print 'Validation data shape: ', X_val.shape
##print 'Validation labels shape: ', y_val.shape
##print 'Test data shape: ', X_test.shape
##print 'Test labels shape: ', y_test.shape

W = np.random.randn(10,3073)*0.0001
##loss, grad = softmax_loss_naive(W, X_train, y_train, 0.0)
##print 'loss: %f' % loss
loss, grad = softmax_loss_vectorized(W, X_train, y_train, 0.0)
print 'loss: %f' % loss

from gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(w, X_train, y_train, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

##import numpy as np
##a = np.arange(15).reshape(3, 5)*0.1
##probs = a / np.sum(a, axis=0)
##y = np.random.choice(3, 5)

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.

results = {}
best_val = -1
best_softmax = None
learning_rates = np.logspace(-10, 10, 10) # np.logspace(-10, 10, 8) #-10, -9, -8, -7, -6, -5, -4
regularization_strengths = np.logspace(-3, 6, 10)

iters = 100

for r in learning_rates:
    for rs in regularization_strengths:
      softmax = Softmax()
      softmax.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=iters)
      y_train_pred = softmax.predict(X_train)
      acc_train = np.mean(y_train == y_train_pred)
      y_val_pred = softmax.predict(X_val)
      acc_val = np.mean(y_val == y_val_pred)

      results[(lr, rs)] = (acc_train, acc_val)

      if best_val < acc_val:
        best_val = acc_val
        best_softmax = softmax

y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)

w = best_softmax.W[:,:-1] # strip out the bias
w = w.reshape(10, 32, 32, 3)

w_min, w_max = np.min(w), np.max(w)

# Visualize the learned weights for each class
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
  plt.subplot(2, 5, i + 1)
  
  # Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
  plt.imshow(wimg.astype('uint8'))
  plt.axis('off')
  plt.title(classes[i])
