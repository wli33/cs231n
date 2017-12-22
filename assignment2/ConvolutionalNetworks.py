#from sklearn.externals import joblib
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from data_utils import *
from vis_utils import visualize_grid

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR+'\\helper')
from solver import Solver
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from layers import *
from fast_layers import *
from data_augmentation import *

sys.path.insert(0, CURRENT_DIR+'\\classifier')
from cnn import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

###Naive forward pass
##x_shape = (2, 3, 4, 4)
##w_shape = (3, 3, 4, 4)
##x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
##w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
##b = np.linspace(-0.1, 0.2, num=3)
##
##conv_param = {'stride': 2, 'pad': 1}
##out, _ = conv_forward_naive(x, w, b, conv_param)
##correct_out = np.array([[[[[-0.08759809, -0.10987781],
##                           [-0.18387192, -0.2109216 ]],
##                          [[ 0.21027089,  0.21661097],
##                           [ 0.22847626,  0.23004637]],
##                          [[ 0.50813986,  0.54309974],
##                           [ 0.64082444,  0.67101435]]],
##                         [[[-0.98053589, -1.03143541],
##                           [-1.19128892, -1.24695841]],
##                          [[ 0.69108355,  0.66880383],
##                           [ 0.59480972,  0.56776003]],
##                          [[ 2.36270298,  2.36904306],
##                           [ 2.38090835,  2.38247847]]]]])
##
### Compare your output to ours; difference should be around 1e-8
##print 'Testing conv_forward_naive'
##print 'difference: ', rel_error(out, correct_out)

###Test Naive backward pass
##x = np.random.randn(4, 3, 5, 5)
##w = np.random.randn(2, 3, 3, 3)
##b = np.random.randn(2,)
##dout = np.random.randn(4, 2, 5, 5)
##conv_param = {'stride': 1, 'pad': 1}
##
##dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
##dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
##db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)
##
##out, cache = conv_forward_naive(x, w, b, conv_param)
##dx, dw, db = conv_backward_naive(dout, cache)
##
### Your errors should be around 1e-9'
##print 'Testing conv_backward_naive function'
##print 'dx error: ', rel_error(dx, dx_num)
##print 'dw error: ', rel_error(dw, dw_num)
##print 'db error: ', rel_error(db, db_num)

###Test Max pooling: Naive forward
##x_shape = (2, 3, 4, 4)
##x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
##pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
##
##out, _ = max_pool_forward_naive(x, pool_param)
##
##correct_out = np.array([[[[-0.26315789, -0.24842105],
##                          [-0.20421053, -0.18947368]],
##                         [[-0.14526316, -0.13052632],
##                          [-0.08631579, -0.07157895]],
##                         [[-0.02736842, -0.01263158],
##                          [ 0.03157895,  0.04631579]]],
##                        [[[ 0.09052632,  0.10526316],
##                          [ 0.14947368,  0.16421053]],
##                         [[ 0.20842105,  0.22315789],
##                          [ 0.26736842,  0.28210526]],
##                         [[ 0.32631579,  0.34105263],
##                          [ 0.38526316,  0.4       ]]]])
##
### Compare your output with ours. Difference should be around 1e-8.
##print 'Testing max_pool_forward_naive function:'
##print 'difference: ', rel_error(out, correct_out)
##

###Test Max pooling: Naive backward
##x = np.random.randn(3, 2, 8, 8)
##dout = np.random.randn(3, 2, 4, 4)
##pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
##
##dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)
##
##out, cache = max_pool_forward_naive(x, pool_param)
##dx = max_pool_backward_naive(dout, cache)
##
### Your error should be around 1e-12
##print 'Testing max_pool_backward_naive function:'
##print 'dx error: ', rel_error(dx, dx_num)

###fast layer implemention

##from time import time

##x = np.random.randn(100, 3, 32, 32)
##dout = np.random.randn(100, 3, 16, 16)
##pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
##
##t0 = time()
##out_naive, cache_naive = max_pool_forward_naive(x, pool_param)
##t1 = time()
##out_fast, cache_fast = max_pool_forward_fast(x, pool_param)
##t2 = time()
##
##print 'Testing pool_forward_fast:'
##print 'Naive: %fs' % (t1 - t0)
##print 'fast: %fs' % (t2 - t1)
##print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
##print 'difference: ', rel_error(out_naive, out_fast)
##
##t0 = time()
##dx_naive = max_pool_backward_naive(dout, cache_naive)
##t1 = time()
##dx_fast = max_pool_backward_fast(dout, cache_fast)
##t2 = time()
##
##print '\nTesting pool_backward_fast:'
##print 'Naive: %fs' % (t1 - t0)
##print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
##print 'dx difference: ', rel_error(dx_naive, dx_fast)
##

#Testing conv_relu_pool_forward
from layer_utils import *

##x = np.random.randn(2, 3, 16, 16)
##w = np.random.randn(3, 3, 3, 3)
##b = np.random.randn(3,)
##dout = np.random.randn(2, 3, 8, 8)
##conv_param = {'stride': 1, 'pad': 1}
##pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
##
##out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)
##dx, dw, db = conv_relu_pool_backward(dout, cache)
##
##dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x, dout)
##dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w, dout)
##db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b, dout)
##
##print 'Testing conv_relu_pool'
##print 'dx error: ', rel_error(dx_num, dx)
##print 'dw error: ', rel_error(dw_num, dw)
##print 'db error: ', rel_error(db_num, db)

###Testing conv_relu_forward
##x = np.random.randn(2, 3, 8, 8)
##w = np.random.randn(3, 3, 3, 3)
##b = np.random.randn(3,)
##dout = np.random.randn(2, 3, 8, 8)
##conv_param = {'stride': 1, 'pad': 1}
##
##out, cache = conv_relu_forward(x, w, b, conv_param)
##dx, dw, db = conv_relu_backward(dout, cache)
##
##dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, w, b, conv_param)[0], x, dout)
##dw_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, w, b, conv_param)[0], w, dout)
##db_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, w, b, conv_param)[0], b, dout)
##
##print 'Testing conv_relu:'
##print 'dx error: ', rel_error(dx_num, dx)
##print 'dw error: ', rel_error(dw_num, dw)
##print 'db error: ', rel_error(db_num, db)

###Sanity check loss
##model = ThreeLayerConvNet()
##
##N = 50
##X = np.random.randn(N, 3, 32, 32)
##y = np.random.randint(10, size=N)
##
##loss, grads = model.loss(X, y)
##print 'Initial loss (no regularization): ', loss
##
##model.reg = 0.5
##loss, grads = model.loss(X, y)
##print 'Initial loss (with regularization): ', loss

###Gradient check
##
##num_inputs = 2
##input_dim = (3, 10, 10)
##reg = 0.0
##num_classes = 10
##X = np.random.randn(num_inputs, *input_dim)
##y = np.random.randint(num_classes, size=num_inputs)
##
##model = ThreeLayerConvNet(num_filters=3, filter_size=3,
##                          input_dim=input_dim, hidden_dim=7,
##                          dtype=np.float64)
##loss, grads = model.loss(X, y)
##for param_name in sorted(grads):
##    f = lambda _: model.loss(X, y)[0]
##    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
##    e = rel_error(param_grad_num, grads[param_name])
##    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
##

###Overfit small data without batchnorm
##data = get_CIFAR10_data()
##num_train = 100
##small_data = {
##  'X_train': data['X_train'][:num_train],
##  'y_train': data['y_train'][:num_train],
##  'X_val': data['X_val'],
##  'y_val': data['y_val'],
##}
##
##model = ThreeLayerConvNet(weight_scale=1e-2)
##
##solver = Solver(model, small_data,
##                num_epochs=20, batch_size=50,
##                update_rule='rmsprop',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=1)
##solver.train()
##
##plt.subplot(2, 1, 1)
##plt.plot(solver.loss_history, 'o')
##plt.xlabel('iteration')
##plt.ylabel('loss')
##
##plt.subplot(2, 1, 2)
##plt.plot(solver.train_acc_history, '-o')
##plt.plot(solver.val_acc_history, '-o')
##plt.legend(['train', 'val'], loc='upper left')
##plt.xlabel('epoch')
##plt.ylabel('accuracy')
##plt.show()

### Visualize Filters
##model = ThreeLayerConvNet(weight_scale=0.002, hidden_dim=100, reg=0.001)
##
##solver = Solver(model, data,
##                num_epochs=2, batch_size=50,
##                update_rule='rmsprop',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=10)
##solver.train()
##
####grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
####plt.imshow(grid.astype('uint8'))
####plt.axis('off')
####plt.gcf().set_size_inches(5, 5)
####plt.show()
##
##img = model.params['W1'].transpose(0, 2, 3, 1)[5]
##low, high = np.min(img), np.max(img)
##img = 255 * (img - low) / (high - low)
##plt.imshow(img.astype('uint8'))
##plt.axis('off')
##plt.gcf().set_size_inches(5, 5)
##plt.show()

###Spatial batch normalization: forward

### Check the training-time forward pass by checking means and variances
### of features both before and after spatial batch normalization

##N, C, H, W = 2, 3, 4, 5
##x = 4 * np.random.randn(N, C, H, W) + 10
##
##print 'Before spatial batch normalization:'
##print '  Shape: ', x.shape
##print '  Means: ', x.mean(axis=(0, 2, 3))
##print '  Stds: ', x.std(axis=(0, 2, 3))
##
### Means should be close to zero and stds close to one
##gamma, beta = np.ones(C), np.zeros(C)
##bn_param = {'mode': 'train'}
##out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
##print 'After spatial batch normalization:'
##print '  Shape: ', out.shape
##print '  Means: ', out.mean(axis=(0, 2, 3))
##print '  Stds: ', out.std(axis=(0, 2, 3))
##
### Means should be close to beta and stds close to gamma
##gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
##out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
##print 'After spatial batch normalization (nontrivial gamma, beta):'
##print '  Shape: ', out.shape
##print '  Means: ', out.mean(axis=(0, 2, 3))
##print '  Stds: ', out.std(axis=(0, 2, 3))

### Check the test-time forward pass by running the training-time
### forward pass many times to warm up the running averages, and then
### checking the means and variances of activations after a test-time
### forward pass.
##
##N, C, H, W = 10, 4, 11, 12
##
##bn_param = {'mode': 'train'}
##gamma = np.ones(C)
##beta = np.zeros(C)
##for t in xrange(50):
##  x = 2.3 * np.random.randn(N, C, H, W) + 13
##  spatial_batchnorm_forward(x, gamma, beta, bn_param)
##bn_param['mode'] = 'test'
##x = 2.3 * np.random.randn(N, C, H, W) + 13
##a_norm, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
##
### Means should be close to zero and stds close to one, but will be
### noisier than training-time forward passes.
##print 'After spatial batch normalization (test-time):'
##print '  means: ', a_norm.mean(axis=(0, 2, 3))
##print '  stds: ', a_norm.std(axis=(0, 2, 3))

###Spatial batch normalization: backward

##N, C, H, W = 2, 3, 4, 5
##x = 5 * np.random.randn(N, C, H, W) + 12
##gamma = np.random.randn(C)
##beta = np.random.randn(C)
##dout = np.random.randn(N, C, H, W)
##
##bn_param = {'mode': 'train'}
##fx = lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
##fg = lambda a: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
##fb = lambda b: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
##
##dx_num = eval_numerical_gradient_array(fx, x, dout)
##da_num = eval_numerical_gradient_array(fg, gamma, dout)
##db_num = eval_numerical_gradient_array(fb, beta, dout)
##
##_, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
##dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
##print 'dx error: ', rel_error(dx_num, dx)
##print 'dgamma error: ', rel_error(da_num, dgamma)
##print 'dbeta error: ', rel_error(db_num, dbeta)

###Testing conv_norm_relu_forward
##
##x = np.random.randn(2, 3, 16, 16)
##w = np.random.randn(3, 3, 3, 3)
##b = np.random.randn(3,)
##gamma = np.ones(3)
##beta = np.ones(3)
##C = 3
##bn_param = {'mode': 'train',
##            'running_mean': np.zeros(C),
##            'running_var': np.zeros(C)}
##
##dout = np.random.randn(2, 3, 16, 16)
##conv_param = {'stride': 1, 'pad': 1}
##
##out, cache = conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)
##dx, dw, db, dgamma, dbeta = conv_norm_relu_backward(dout, cache)
##
##dx_num = eval_numerical_gradient_array(lambda x: conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], x, dout)
##dw_num = eval_numerical_gradient_array(lambda w: conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], w, dout)
##db_num = eval_numerical_gradient_array(lambda b: conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], b, dout)
##dgamma_num = eval_numerical_gradient_array(lambda gamma:conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], gamma, dout)
##dbeta_num = eval_numerical_gradient_array(lambda beta:conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], beta, dout)
##
##
##print 'Testing conv_norm_relu_pool'
##print 'dx error: ', rel_error(dx_num, dx)
##print 'dw error: ', rel_error(dw_num, dw)
##print 'db error: ', rel_error(db_num, db)
##print 'dbeta error: ', rel_error(dbeta_num, dbeta)
##print 'dgamma error: ', rel_error(dgamma_num, dgamma)

###Testing conv_norm_relu_pool_forward
##
##x = np.random.randn(2, 3, 16, 16)
##w = np.random.randn(3, 3, 3, 3)
##b = np.random.randn(3,)
##gamma = np.ones(3)
##beta = np.ones(3)
##bn_param = {'mode': 'train',
##            'running_mean': np.zeros(3),
##            'running_var': np.zeros(3)}
##
##dout = np.random.randn(2, 3, 8, 8)
##conv_param = {'stride': 1, 'pad': 1}
##pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
##
##out, cache = conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)
##dx, dw, db, dgamma, dbeta = conv_norm_relu_pool_backward(dout, cache)
##
##dx_num = eval_numerical_gradient_array(lambda x: conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], x, dout)
##dw_num = eval_numerical_gradient_array(lambda w: conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], w, dout)
##db_num = eval_numerical_gradient_array(lambda b: conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], b, dout)
##dgamma_num = eval_numerical_gradient_array(lambda gamma:conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], gamma, dout)
##dbeta_num = eval_numerical_gradient_array(lambda beta:conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], beta, dout)
##
##
##print 'Testing conv_relu_pool'
##print 'dx error: ', rel_error(dx_num, dx)
##print 'dw error: ', rel_error(dw_num, dw)
##print 'db error: ', rel_error(db_num, db)
##print 'dbeta error: ', rel_error(dbeta_num, dbeta)
##print 'dgamma error: ', rel_error(dgamma_num, dgamma)

###Sanity check loss
##model = ThreeLayerConvNet(use_batchnorm = True)
##
##N = 50
##X = np.random.randn(N, 3, 32, 32)
##y = np.random.randint(10, size=N)
##
##loss, grads = model.loss(X, y)
##print 'Initial loss (no regularization): ', loss
##
##model.reg = 0.5
##loss, grads = model.loss(X, y)
##print 'Initial loss (with regularization): ', loss

###Sanity check 2 : Gradient check
##
##num_inputs = 2
##input_dim = (3, 10, 10)
##reg = 0.0
##num_classes = 10
##X = np.random.randn(num_inputs, *input_dim)
##y = np.random.randint(num_classes, size=num_inputs)
##
##model = ThreeLayerConvNet(num_filters=3, filter_size=3,
##                          input_dim=input_dim, hidden_dim=7,
##                          dtype=np.float64,use_batchnorm = True)
##loss, grads = model.loss(X, y)
##for param_name in sorted(grads):
##    f = lambda _: model.loss(X, y)[0]
##    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
##    e = rel_error(param_grad_num, grads[param_name])
##    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

data = get_CIFAR10_data(num_training = 10,normalize = False)

###Sanity check 3: Overfit small subset of the data with batchnorm
##num_train = 100
##small_data = {
##  'X_train': data['X_train'][:num_train],
##  'y_train': data['y_train'][:num_train],
##  'X_val': data['X_val'],
##  'y_val': data['y_val'],
##}
##
##model = ThreeLayerConvNet(weight_scale=2e-2,use_batchnorm = True)
##
##solver = Solver(model, small_data,
##                num_epochs=10, batch_size=50,
##                update_rule='rmsprop',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=2)
##solver.train()
##
##plt.subplot(2, 1, 1)
##plt.plot(solver.loss_history, 'o')
##plt.xlabel('iteration')
##plt.ylabel('loss')
##
##plt.subplot(2, 1, 2)
##plt.plot(solver.train_acc_history, '-o')
##plt.plot(solver.val_acc_history, '-o')
##plt.legend(['train', 'val'], loc='upper left')
##plt.xlabel('epoch')
##plt.ylabel('accuracy')
##plt.show()

##model_base = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=30, reg=0.001)
##
##solver_base = Solver(model_base, data,
##                num_epochs=1, batch_size=50,
##                update_rule='adam',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=50)
##solver_base.train()

##model_1 = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=30, reg=0.001, filter_size = 3)
##
##solver_1 = Solver(model_1, data,
##                num_epochs=1, batch_size=50,
##                update_rule='rmsprop',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=50)
##solver_1.train()

##model_2 = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=50, reg=0.001,
##                            filter_size = 3,use_batchnorm = True)
##
##solver_2 = Solver(model_2, data,
##                num_epochs=1, batch_size=50,
##                update_rule='rmsprop',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=50)
##solver_2.train()
##
##model_3 = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=30, reg=0.001,
##                            filter_size = 3,num_filters = 45)
##
##solver_3 = Solver(model_3, data,
##                num_epochs=1, batch_size=50,
##                update_rule='adam',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=50)
##solver_3.train()
##
##model_4 = FirstConvNet(weight_scale=0.001,reg=0.001)
##
##solver_4 = Solver(model_4, data,
##                num_epochs=1, batch_size=50,
##                update_rule='adam',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=50)
##solver_4.train()

##data_augumentation(data)
X = data['X_train']
num_imgs = 8
X = X[np.random.randint(10, size=num_imgs)]

X_flip = random_flips(X)
X_rand_crop = random_crops(X, (28, 28))

# To give more dramatic visualizations we use large scales for random contrast
# and tint adjustment.
X_contrast = random_contrast(X, scale=(0.5, 1.0))
X_tint = random_tint(X, scale=(-50, 50))

next_plt = 1
for i in xrange(num_imgs):
    titles = ['original', 'flip', 'rand crop', 'contrast', 'tint']
    for j, XX in enumerate([X, X_flip, X_rand_crop, X_contrast, X_tint]):
        plt.subplot(num_imgs, 5, next_plt)
        img = XX[i].transpose(1, 2, 0)
        if j == 4:
            # For visualization purposes we rescale the pixel values of the
            # tinted images
            low, high = np.min(img), np.max(img)
            img = 255 * (img - low) / (high - low)
        plt.imshow(img.astype('uint8'))
        if i == 0:
            plt.title(titles[j])
        plt.gca().axis('off')
        next_plt += 1
plt.show()
##
##model_5 = FirstConvNet(weight_scale=0.001,reg=0.001)
##
##solver_5 = Solver(model_5, data,
##                num_epochs=1, batch_size=50,
##                update_rule='rmsprop',
##                optim_config={
##                  'learning_rate': 1e-4,
##                },
##                verbose=True, print_every=50)
##solver_5.train()

##import seaborn as sns
##def plot_loss(solvers):
##    fig = plt.figure(figsize = (20,10))
##    ax = plt.subplot(111)
##    for key,solver in solvers.iteritems():
##        ax.plot(solver.loss_history, label =key)
##        
##    ax.legend(fontsize = 24)
##    ax.tick_params(labelsize = 24)
##    sns.plt.show()
    
##def plot(solvers):
##    plt.subplot(3, 1, 1)
##    plt.title('Training loss')
##    plt.xlabel('Iteration')
##
##    plt.subplot(3, 1, 2)
##    plt.title('Training accuracy')
##    plt.xlabel('Epoch')
##
##    plt.subplot(3, 1, 3)
##    plt.title('Validation accuracy')
##    plt.xlabel('Epoch')
##
##    for update_rule, solver in solvers.iteritems():
##      plt.subplot(3, 1, 1)
##      plt.plot(solver.loss_history, 'o', label=update_rule)
##      
##      plt.subplot(3, 1, 2)
##      plt.plot(solver.train_acc_history, '-o', label=update_rule)
##
##      plt.subplot(3, 1, 3)
##      plt.plot(solver.val_acc_history, '-o', label=update_rule)
##      
##    for i in [1, 2, 3]:
##      plt.subplot(3, 1, i)
##      plt.legend(loc='upper center', ncol=4)
##      
##    plt.gcf().set_size_inches(15, 15)
##    plt.show()
##        
##solvers = {'solver_base':solver_base,
##          #'smallfilter': solver_1,
##          #'Batchnorm': solver_2,
##          'morefilters': solver_3,
##           'L-layer':solver_4,
##          #'Hflip': solver_5
##          }
##plot(solvers)
