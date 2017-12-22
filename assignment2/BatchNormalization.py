import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data

import sys,os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR+'\\helper')
from solver import Solver
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

sys.path.insert(0, CURRENT_DIR+'\\classifier')
from fc_net import *
#from solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

### Check the training-time forward pass by checking means and variances
### of features both before and after batch normalization
##
### Simulate the forward pass for a two-layer network
##N, D1, D2, D3 = 200, 50, 60, 3
##X = np.random.randn(N, D1)
##W1 = np.random.randn(D1, D2)
##W2 = np.random.randn(D2, D3)
##a = np.maximum(0, X.dot(W1)).dot(W2)
##
##print 'Before batch normalization:'
##print '  means: ', a.mean(axis=0)
##print '  stds: ', a.std(axis=0)
##
### Means should be close to zero and stds close to one
##print 'After batch normalization (gamma=1, beta=0)'
##a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
##print '  mean: ', a_norm.mean(axis=0)
##print '  std: ', a_norm.std(axis=0)
##
### Now means should be close to beta and stds close to gamma
##gamma = np.asarray([1.0, 2.0, 3.0])
##beta = np.asarray([11.0, 12.0, 13.0])
##a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
##print 'After batch normalization (nontrivial gamma, beta)'
##print '  means: ', a_norm.mean(axis=0)
##print '  stds: ', a_norm.std(axis=0)

# Check the test-time forward pass by running the training-time
# forward pass many times to warm up the running averages, and then
# checking the means and variances of activations after a test-time
# forward pass.

##N, D1, D2, D3 = 200, 50, 60, 3
##W1 = np.random.randn(D1, D2)
##W2 = np.random.randn(D2, D3)
##
##bn_param = {'mode': 'train'}
##gamma = np.ones(D3)
##beta = np.zeros(D3)
##for t in xrange(50):
##  X = np.random.randn(N, D1)
##  a = np.maximum(0, X.dot(W1)).dot(W2)
##  batchnorm_forward(a, gamma, beta, bn_param)
##bn_param['mode'] = 'test'
##X = np.random.randn(N, D1)
##a = np.maximum(0, X.dot(W1)).dot(W2)
##a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)
##
### Means should be close to zero and stds close to one, but will be
### noisier than training-time forward passes.
##print 'After batch normalization (test-time):'
##print '  means: ', a_norm.mean(axis=0)
##print '  stds: ', a_norm.std(axis=0)

# Gradient check batchnorm backward pass

##N, D = 4, 5
##x = 5 * np.random.randn(N, D) + 12
##gamma = np.random.randn(D)
##beta = np.random.randn(D)
##dout = np.random.randn(N, D)
##
##bn_param = {'mode': 'train'}
##fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
##fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
##fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]
##
##dx_num = eval_numerical_gradient_array(fx, x, dout)
##da_num = eval_numerical_gradient_array(fg, gamma, dout)
##db_num = eval_numerical_gradient_array(fb, beta, dout)
##
##_, cache = batchnorm_forward(x, gamma, beta, bn_param)
##dx, dgamma, dbeta = batchnorm_backward(dout, cache)
##print 'dx error: ', rel_error(dx_num, dx)
##print 'dgamma error: ', rel_error(da_num, dgamma)
##print 'dbeta error: ', rel_error(db_num, dbeta)

## Test simple example
##N, D = 4, 5
##x = 5 * np.random.randn(N, D) + 12
##gamma = np.random.randn(D)
##beta = np.random.randn(D)
##dout = np.random.randn(N, D)
##
##def forward(x):
##    #Step 1
##    N,D = x.shape
##    mu = 1 / float(N) * np.sum(x, axis=0)
##    #Step 2
##    xcenter = x - mu
##    #Step 3
##    out = xcenter
##    cache = (mu,xcenter)
##    
##    return out,cache
##
###Backward pass
##def backward(dout,cache):
##    mu,xcenter = cache
##    #Backprop
##    #Step 3
##    dxcenter = dout
##
##    #Step2
##    dx = np.ones(x.shape)*dxcenter
##    dmu = -np.sum(dxcenter,0)
##
##    #Step1
##    dx += 1/float(N)*np.ones(x.shape)*dmu 
##    return dx
##fx = lambda x: forward(x)[0]
##
##dx_num = eval_numerical_gradient_array(fx, x, dout)
##
##_, cache = forward(x)
##dx = backward(dout, cache)
##print 'dx error: ', rel_error(dx_num, dx)
##
##def backward_alt(dout,cache):
##    mu,xcenter = cache
##    #Backprop
##    #Step 3
##    dx = dout-1./float(N)*np.sum(dout,axis=0)
##    return dx
##dx2 = backward_alt(dout,cache)
##rel_error(dx2,dx)

##N, D = 100, 500
##x = 5 * np.random.randn(N, D) + 12
##gamma = np.random.randn(D)
##beta = np.random.randn(D)
##dout = np.random.randn(N, D)
##
##bn_param = {'mode': 'train'}
##out, cache = batchnorm_forward(x, gamma, beta, bn_param)
##
##import time
##t1 = time.time()
##dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)
##t2 = time.time()
##dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)
##t3 = time.time()
##
##print 'dx difference: ', rel_error(dx1, dx2)
##print 'dgamma difference: ', rel_error(dgamma1, dgamma2)
##print 'dbeta difference: ', rel_error(dbeta1, dbeta2)
###print 'speedup: %.2fx' % ((t2 - t1) / (t3 - t2))

###Test Batch-normal with small data
##N, D, H1, H2, C = 2, 10, 20, 30, 10
##X = np.random.randn(N, D)
##y = np.random.randint(C, size=(N,))
##
##for reg in [0, 3.14]:
##  print 'Running check with reg = ', reg
##  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
##                            reg=reg, weight_scale=5e-2, dtype=np.float64,
##                            use_batchnorm=True)
##
##  loss, grads = model.loss(X, y)
##  print 'Initial loss: ', loss
##
##  for name in sorted(grads):
##    f = lambda _: model.loss(X, y)[0]
##    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
##    print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
##  if reg == 0: print

###Batchnorm for deep networks
data = get_CIFAR10_data()
### Try training a very deep net with batchnorm
##hidden_dims = [100, 100, 100, 100 ,100]
##
##num_train = 1000
##small_data = {
##  'X_train': data['X_train'][:num_train],
##  'y_train': data['y_train'][:num_train],
##  'X_val': data['X_val'],
##  'y_val': data['y_val'],
##}
##
##weight_scale = 2e-2
##bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
##model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)
##
##bn_solver = Solver(bn_model, small_data,
##                num_epochs=30, batch_size=50,
##                update_rule='rmsprop',
##                optim_config={
##                  'learning_rate': 1e-3,
##                },
##                verbose=True, print_every=200)
##bn_solver.train()
##
##solver = Solver(model, small_data,
##                num_epochs=30, batch_size=50,
##                update_rule='rmsprop',
##                optim_config={
##                  'learning_rate': 1e-3,
##                },
##                verbose=True, print_every=200)
##solver.train()
##plt.subplot(3, 1, 1)
##plt.title('Training loss')
##plt.xlabel('Iteration')
##
##plt.subplot(3, 1, 2)
##plt.title('Training accuracy')
##plt.xlabel('Epoch')
##
##plt.subplot(3, 1, 3)
##plt.title('Validation accuracy')
##plt.xlabel('Epoch')
##
##plt.subplot(3, 1, 1)
##plt.plot(solver.loss_history, 'o', label='baseline')
##plt.plot(bn_solver.loss_history, 'o', label='batchnorm')
##
##plt.subplot(3, 1, 2)
##plt.plot(solver.train_acc_history, '-o', label='baseline')
##plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')
##
##plt.subplot(3, 1, 3)
##plt.plot(solver.val_acc_history, '-o', label='baseline')
##plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')
##  
##for i in [1, 2, 3]:
##  plt.subplot(3, 1, i)
##  plt.legend(loc='upper center', ncol=4)
##plt.gcf().set_size_inches(15, 15)
##plt.show()

# Try training a very deep net with batchnorm
hidden_dims = [50, 50, 50, 50, 50, 50, 50]

num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

bn_solvers = {}
solvers = {}
weight_scales = np.logspace(-4, 0, num=15)
for i, weight_scale in enumerate(weight_scales):
  print 'Running weight scale %d / %d' % (i + 1, len(weight_scales))
  bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
  model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)

  bn_solver = Solver(bn_model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='rmsprop',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  verbose=False, print_every=200)
  bn_solver.train()
  bn_solvers[weight_scale] = bn_solver

  solver = Solver(model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='rmsprop',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  verbose=False, print_every=200)
  solver.train()
  solvers[weight_scale] = solver

# Plot results of weight scale experiment
best_train_accs, bn_best_train_accs = [], []
best_val_accs, bn_best_val_accs = [], []
final_train_loss, bn_final_train_loss = [], []

for ws in weight_scales:
  best_train_accs.append(max(solvers[ws].train_acc_history))
  bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))
  
  best_val_accs.append(max(solvers[ws].val_acc_history))
  bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))
  
  final_train_loss.append(np.mean(solvers[ws].loss_history))
  bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history))
  
plt.subplot(3, 1, 1)
plt.title('Best val accuracy vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Best val accuracy')
plt.semilogx(weight_scales, best_val_accs, '-o', label='baseline')
plt.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')
plt.legend(ncol=2, loc='lower right')

plt.subplot(3, 1, 2)
plt.title('Best train accuracy vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Best training accuracy')
plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
plt.legend()

plt.subplot(3, 1, 3)
plt.title('Final training loss vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Final training loss')
plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
plt.legend()

plt.gcf().set_size_inches(10, 15)
plt.show()
