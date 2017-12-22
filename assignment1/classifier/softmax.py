import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1] # d*n
  num_class = W.shape[0]

  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.           #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0.0
  for i in range(num_train):
      X_i = X[:,i] # D*1
      score_i = W.dot(X_i)
      score_i -= np.max(score_i) #C*1 but keepdims = false so it becomes 1*C
      exp_score_i = np.exp(score_i)
      probs_i = exp_score_i/np.sum(exp_score_i) #1*C
      correct_logprobs_i = -np.log(probs_i[y[i]])
      loss += correct_logprobs_i
      
      dscore_i = probs_i.reshape(num_class,-1)#c*1
      dscore_i[y[i]] -= 1 #C*1
      X_i = X_i.reshape(1,-1)# 1*D
      dW += dscore_i.dot(X_i)
      
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)

  dW /= num_train
  dW += reg*W
  
  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  m = X.shape[1]
  f = W.dot(X)
  f -= np.max(f,axis = 0)
  exp_score = np.exp(f)
  probs = exp_score/np.sum(exp_score,axis = 0)
  corect_logprobs = -np.log(probs[y,range(m)])
  loss = np.sum(corect_logprobs)/m
  
  loss += 0.5*reg*np.sum(W*W)

  dscore = probs
  dscore[y,range(m)] -= 1 #C*N

  dW = np.dot(dscore,X.T)/m #x.T:n*d
  dW+= reg*W

  return loss, dW
