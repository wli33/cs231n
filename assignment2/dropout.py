import time
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

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

###Dropout forward pass
##
##x = np.random.randn(500, 500) + 10
##
##for p in [0.3, 0.6, 0.1]:
##  out, cache = dropout_forward(x, {'mode': 'train', 'p': p})
##  out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})
##
##  print 'Running tests with p = ', p
##  print 'Mean of input: ', x.mean()
##  print 'Mean of train-time output: ', out.mean()
##  print 'Mean of test-time output: ', out_test.mean()
##  print 'Fraction of train-time output set to zero: ', (out == 0).mean()
##  print 'Fraction of test-time output set to zero: ', (out_test == 0).mean()
##  print

###Dropout backward pass
##x = np.random.randn(10, 10) + 10
##dout = np.random.randn(*x.shape)
##
##dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}
##out, cache = dropout_forward(x, dropout_param)
##dx = dropout_backward(dout, cache)
##dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)
##
##print 'dx relative error: ', rel_error(dx, dx_num)

### Test fc_net with dropout
##N, D, H1, H2, C = 2, 15, 20, 30, 10
##X = np.random.randn(N, D)
##y = np.random.randint(C, size=(N,))
##
##for dropout in [0, 0.25, 1.0]:
##  print 'Running check with dropout = ', dropout
##  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
##                            weight_scale=5e-2, dtype=np.float64,
##                            dropout=dropout, seed=123)
##
##  loss, grads = model.loss(X, y)
##  print 'Initial loss: ', loss
##
##  for name in sorted(grads):
##    f = lambda _: model.loss(X, y)[0]
##    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
##    print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
##  print

#Regularization experiment
data = get_CIFAR10_data()

num_train = 500
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}
dropout_choices = [0.0, 0.75]
for dropout in dropout_choices:
  model = FullyConnectedNet([500], dropout=dropout)
  print dropout

  solver = Solver(model, small_data,
                  num_epochs=15, batch_size=100,
                  update_rule='rmsprop',
                  optim_config={
                    'learning_rate': 5e-4,
                  },
                  verbose=True, print_every=200)
  solver.train()
  solvers[dropout] = solver

# Plot train and validation accuracies of the two models

train_accs = []
val_accs = []
for dropout in dropout_choices:
  solver = solvers[dropout]
  train_accs.append(solver.train_acc_history[-1])
  val_accs.append(solver.val_acc_history[-1])

plt.subplot(3, 1, 1)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
plt.title('Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
  
plt.subplot(3, 1, 2)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
plt.title('Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')

plt.gcf().set_size_inches(15, 15)
plt.show()
