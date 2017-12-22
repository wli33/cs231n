import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, sys.path[0]+'\\classifier')
from KNearestNeighbor import KNearestNeighbor

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

cifar10_dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

### As a sanity check, we print out the size of the training and test data.
##print 'Training data shape: ', X_train.shape
##print 'Training labels shape: ', y_train.shape
##print 'Test data shape: ', X_test.shape
##print 'Test labels shape: ', y_test.shape

num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

##import numpy as np
##a = np.arange(15).reshape(3, 5)
##b = np.arange(10).reshape(2, 5)
##tr = np.square(a).sum(axis = 1) #per instance
##te = np.square(b).sum(axis = 1)
##M = np.dot(b,a.T)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
#print X_train.shape, X_test.shape


# CV: Just saves the data

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
dists = classifier.compute_distances_no_loops(X_test)
#plt.imshow(dists, interpolation='none')
#plt.savefig('sample.png')

y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred== y_test)
accuracy = float(num_correct)/num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
##dists_one = classifier.compute_distances_one_loop(X_test)
##
### To ensure that our vectorized implementation is correct, we make sure that it
### agrees with the naive implementation. There are many ways to decide whether
### two matrices are similar; one of the simplest is the Frobenius norm. In case
### you haven't seen it before, the Frobenius norm of two matrices is the square
### root of the squared sum of differences of all elements; in other words, reshape
### the matrices into vectors and compute the Euclidean distance between them.
##difference = np.linalg.norm(dists - dists_one, ord='fro')
##print 'Difference was: %f' % (difference, )
##if difference < 0.001:
##  print 'Good! The distance matrices are the same'
##else:
##  print 'Uh-oh! The distance matrices are different'

##  
##
##def time_function(f, *args):
##  """
##  Call a function f with args and return the time (in seconds) that it took to execute.
##  """
##  import time
##  tic = time.time()
##  f(*args)
##  toc = time.time()
##  return toc - tic
##
##two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
##print 'Two loop version took %f seconds' % two_loop_time
##
##one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
##print 'One loop version took %f seconds' % one_loop_time
##
##no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
##print 'No loop version took %f seconds' % no_loop_time

# you should see significantly faster performance with the fully vectorized implementation

num_folds = 5
k_choices = [1,3,5,8,10,12]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train,num_folds)
y_train_folds = np.array_split(y_train,num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []

for k in k_choices:
    for j in range(num_folds):
        X_train_cv = np.vstack(X_train_folds[0:j]+X_train_folds[j+1:])
        X_test_cv = X_train_folds[j]

        y_train_cv = np.hstack(y_train_folds[0:j] +y_train_folds[j+1:])
        y_test_cv = y_train_folds[j]

        classifier.train(X_train_cv,y_train_cv)
        dists_cv = classifier.compute_distances_no_loops(X_test_cv)
        y_test_pred = classifier.predict_labels(dists_cv, k)
        accuracy = np.mean(y_test_pred== y_test)
        k_to_accuracies[k].append(accuracy)

for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)
  
accuracies_mean = [np.mean(v) for k,v in k_to_accuracies.items()]
accuracies_std = [np.std(v) for k,v in k_to_accuracies.items()]
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data.
best_k = 7

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

