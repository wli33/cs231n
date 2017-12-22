import numpy as np

import sys,os
Parent_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Parent_DIR+'\\helper')

from layers import *
from fast_layers import *
from layer_utils import *


class FirstConvNet(object):
    """
    A L-layer convolutional network with the following architecture:
    [conv-relu-pool2x2]xL - [affine - relu]xM - affine - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 32], filter_size=3,
                 hidden_dims=[100, 100], num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: List of size  Nbconv+1 with the number of filters
        to use in each convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dims: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}

        self.L = len(num_filters)
        self.M = len(hidden_dims)
        self.num_layers = 1 + len(hidden_dims)
        self.filter_size = filter_size

        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

       #conv layers
        C,H,W = input_dim
        F = [C] + num_filters
        for i in range(self.L):
            self.params['W%d'%(i+1)] = weight_scale * np.random.randn(F[i+1], F[i],
                                                           filter_size, filter_size)
            self.params['b%d'%(i+1)] = np.zeros(num_filters[i])

            if self.use_batchnorm:
                self.bn_params['gamma%d'%(i+1)] = np.ones(num_filters[i])
                self.bn_params['beta%d'%(i+1)] = np.zeros(num_filters[i])

                self.bn_params['bn_param%d'%(i+1)] = {'mode': 'train',
                            'running_mean': np.zeros(num_filters[i]),
                            'running_var': np.zeros(num_filters[i])}
        # affine-relu layers
        Hp,Wp = self.size_Conv(H,W,self.L)
        dims = [Hp * Wp * F[-1]] + hidden_dims

        for i in range(self.M):
            idx = self.L + i +1
            self.params['W%d'% idx] = weight_scale * np.random.randn(dims[i],dims[i+1])
            self.params['b%d'% idx] = np.zeros(hidden_dims[i])

            if self.use_batchnorm:
                self.bn_params['gamma%d'% idx] = np.ones(hidden_dims[i])
                self.bn_params['beta%d'% idx] = np.zeros(hidden_dims[i])
                self.bn_params['bn_param%d'% idx] = {'mode': 'train',
                            'running_mean': np.zeros(hidden_dims[i]),
                            'running_var': np.zeros(hidden_dims[i])}
                
        #score layer
        idx += 1
        self.params['W%d'% idx] =  weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params['b%d'% idx] = np.zeros(num_classes)
        
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def size_Conv(self, H, W, Nbconv):
        P = self.conv_param['pad'] # padd
        S = self.conv_param['stride']
        Hc = (H + 2 * P - self.filter_size) / S + 1
        Wc = (W + 2 * P - self.filter_size) / S + 1
        pool_width = self.pool_param['pool_width']
        pool_height = self.pool_param['pool_height']
        stride_pool = self.pool_param['stride']
        Hp = (Hc - pool_height) / stride_pool + 1
        Wp = (Wc - pool_width) / stride_pool + 1
        if Nbconv == 1:
            return Hp, Wp
        else:
            H = Hp
            W = Wp
            return self.size_Conv(H, W, Nbconv - 1)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        N = X.shape[0]

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size

        # pass pool_param to the forward pass for the max-pooling layer

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                if key[:2] == 'bn':
                    bn_param['mode'] = mode

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #######################################################################
        
        #forward pass
        layers = {}
        layers[0] = X
        cache_layers = {}

        # conv-relu-pool layers
        for i in range(self.L):
            w, b = self.params['W%d'%(i+1)], self.params['b%d'%(i+1)]

            if self.use_batchnorm:
                beta,gamma = self.bn_params['beta%d'%(i+1)],self.bn_params['gamma%d'%(i+1)]
                bn_param = self.bn_params['bn_param%d'%(i+1)]
                layers[i+1],cache_layers[i] = conv_norm_relu_pool_forward(layers[i],
                            w, b, self.conv_param, self.pool_param, gamma, beta, bn_param)
            else:
                layers[i+1],cache_layers[i] = conv_relu_pool_forward(
                    layers[i], w, b, self.conv_param, self.pool_param)
        # affine-relu layers
        for i in xrange(self.M):
            idx = self.L + i + 1
            # convert to FC-layers
            if i == 0:
                layer_beforeFC = layers[idx-1]# preserve the shape for backward pass
                layers[idx-1] = layers[idx-1].reshape(N,-1)
                
            w, b = self.params['W%d'%idx], self.params['b%d'% idx]

            if self.use_batchnorm:
                beta = self.bn_params['beta' + str(idx)]
                gamma = self.bn_params['gamma' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]
                layers[idx], cache_layers[idx-1] = affine_norm_relu_forward(layers[idx-1], w, b, gamma,
                                                      beta, bn_param)
            else:
                layers[idx], cache_layers[idx-1] = affine_relu_forward(layers[idx-1], w, b)

        #affine 
        idx += 1
        w, b = self.params['W%d'%idx], self.params['b%d'% idx]
        layers[idx], cache_layers[idx-1] = affine_forward(layers[idx-1], w, b)
        scores = layers[idx]

        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #######################################################################
        loss,dscore = softmax_loss(scores,y)

        for i in range(self.num_layers):
            loss += 0.5* self.reg *np.sum(self.params['W%d'%(i+1)]**2)

        # Backward pass
        dx = {}
        idx = self.L + self.M
        dx[idx], dw, db = affine_backward(dscore, cache_layers[idx])
        grads['W%d' %(idx+1)],grads['b%d' %(idx+1)] = dw,db
        grads['W%d'% (idx+1)]+= self.reg * self.params['W%d' % (idx+1)]

        for i in range(self.M)[::-1]:
            idx = self.L + i + 1
            if self.use_batchnorm:
                dx[idx-1], dw, db, dgamma, dbeta = affine_norm_relu_backward(
                    dx[idx], cache_layers[idx-1])
                grads['dbeta' + str(idx)] = dbeta
                grads['dgamma' + str(idx)] = dgamma
                pass
            else:
                dx[idx-1], dw, db = affine_relu_backward(dx[idx], cache_layers[idx-1])

            grads['W%d' %(idx)],grads['b%d' %(idx)] = dw,db
            grads['W%d'%(idx)]+= self.reg * self.params['W%d' % idx]

         # Backprop into the conv blocks
        for i in range(self.L)[::-1]:
            idx = i + 1
            if i == max(range(self.L)[::-1]):
                dx[idx] = dx[idx].reshape(*layer_beforeFC.shape)
                
            if self.use_batchnorm:
                dx[i], dw, db, dgamma, dbeta = conv_norm_relu_pool_backward(
                    dx[idx], cache_layers[i])
                grads['beta' + str(idx)] = dbeta
                grads['gamma' + str(idx)] = dgamma
            else:
                dx[i], dw, db = conv_relu_pool_backward(dx[idx], cache_layers[i])

            
            grads['W%d' %(idx)],grads['b%d' %(idx)] = dw,db

            #Regularization
            grads['W%d'%(idx)]+= self.reg * self.params['W%d' % idx]
       

        return loss, grads


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}

        #######################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        #######################################################################
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        self.pool_param = {'pool_width':2,'pool_height':2,'stride':2}


        C,H,W = input_dim

        #input layer
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C,
                                                           filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        
        # Conv layer
        # Input size : (N,C,H,W)
        # Output size : (N,F,Hc,Wc)
        P = self.conv_param['pad']
        S = self.conv_param['stride']
        
        Hc = (H + 2 * P - filter_size) / S + 1
        Wc = (W + 2 * P - filter_size) / S + 1

        #pool layer: 2*2
        # Input : (N,F,Hc,Wc)
        #no parameter
        # Ouput : (N,F,Hp,Wp)

        width_pool = self.pool_param['pool_width']
        height_pool = self.pool_param['pool_height']
        stride_pool = self.pool_param['stride']

        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1

        # FC hidden affine layer
        #input:(N,F*Hp*Wp)
        #parameter:(F*Hp*Wp,Hh)
        #output:(N,Hh)

        self.params['W2'] = weight_scale * np.random.randn(num_filters* Hp * Wp,hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # output affine layer
        # input:(N,Hh)
        # parameter:(Hh,C)
        # output(N,C)

        self.params['W3'] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params['b3'] = np.zeros(num_classes)

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.

        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(num_filters)
            self.params['beta1'] = np.zeros(num_filters)
            self.params['gamma2'] = np.ones(hidden_dim)
            self.params['beta2'] = np.zeros(hidden_dim)
            self.bn_params = {'bn_param0':{'mode': 'train',
                     'running_mean': np.zeros(num_filters),
                     'running_var': np.zeros(num_filters)},
                              'bn_param1':{'mode': 'train',
                     'running_mean': np.zeros(hidden_dim),
                     'running_var': np.zeros(hidden_dim)}
                     }


        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #######################################################################
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        if self.use_batchnorm:
            bn_param1, gamma1, beta1 = self.bn_params['bn_param0'], \
                                       self.params['gamma1'], self.params['beta1']
            bn_param2, gamma2, beta2 = self.bn_params['bn_param1'], \
                                       self.params['gamma2'], self.params['beta2']
            
            X1, cache1 = conv_norm_relu_pool_forward(X, W1, b1,self.conv_param,
                                            self.pool_param, gamma1, beta1, bn_param1)
            X2, cache2 = affine_norm_relu_forward(X1, W2, b2, gamma2, beta2, bn_param2)
            
        else:
            X1,cache_conv_relu_pool = conv_relu_pool_forward(X, W1, b1, self.conv_param,
                                                         self.pool_param)
            X2, cache_affine_relu = affine_relu_forward(X1, W2, b2)
            
        scores, cache_scores = affine_forward(X2, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #######################################################################
        loss,dscores = softmax_loss(scores,y)
        loss += 0.5 * self.reg*sum(np.sum(W**2) for W in [W1, W2, W3])

        da2,dw3,db3 = affine_backward(dscores,cache_scores)
        
        if self.use_batchnorm:
            da1,dw2,db2,dgamma2,dbeta2 = affine_norm_relu_backward(da2,cache2)
            dx,dw1,db1,dgamma1, dbeta1 = conv_norm_relu_pool_backward(da1,cache1)
        else:
            da1,dw2,db2 = affine_relu_backward(da2,cache_affine_relu)
            dx,dw1,db1 = conv_relu_pool_backward(da1,cache_conv_relu_pool)

        dw1 += self.reg*W1
        dw2 += self.reg*W2
        dw3 += self.reg*W3

        grads = {'W1': dw1, 'b1': db1, 'W2': dw2, 'b2': db2,\
                 'W3': dw3, 'b3': db3}
        if self.use_batchnorm:
            grads.update({'beta1': dbeta1,
                          'beta2': dbeta2,
                          'gamma1': dgamma1,
                          'gamma2': dgamma2})
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads


pass
