import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 49000
NUM_VAL = 1000

cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                          transform=T.ToTensor())
loader_test = DataLoader(cifar10_test, batch_size=64)

dtype = torch.FloatTensor # the CPU datatype

# Constant to control how frequently we print train loss
print_every = 100

# This is a little utility that we'll use to reset the model
# if we want to re-initialize all our parameters
def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

#Example Model

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

#The example model itself
# Here's where we define the architecture of the model... 
simple_model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                Flatten(), # see above for explanation
                nn.Linear(5408, 10), # affine layer
              )

# Set the type of all data in this model to be FloatTensor 
simple_model.type(dtype)

loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(simple_model.parameters(), lr=1e-2) # lr sets the learning rate of the optimizer

#Training a specific model
fixed_model_base = nn.Sequential(
                   nn.Conv2d(3, 32, kernel_size=7),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm2d(32),
                   nn.MaxPool2d(2,stride = 2),
                   Flatten(), # see above for explanation
                   nn.Linear(5408,1024),
                   nn.ReLU(inplace=True),
                   nn.Linear(1024, 10), # affine layer
                    )

fixed_model = fixed_model_base.type(dtype)

loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.RMSprop(fixed_model.parameters(), lr=1e-2)

#### Now we're going to feed a random batch into the model you defined and make sure the output is the right size
##x = torch.randn(64, 3, 32, 32).type(dtype)
##x_var = Variable(x.type(dtype)) # Construct a PyTorch Variable out of your input data
##ans = fixed_model(x_var)        # Feed it through the model! 
##
### Check to make sure what comes out of your model
### is the right dimensionality... this should be True
### if you've done everything correctly
##np.array_equal(np.array(ans.size()), np.array([64, 10]))

#GPU
torch.cuda.is_available()
import copy
gpu_dtype = torch.cuda.FloatTensor

fixed_model_gpu = copy.deepcopy(fixed_model_base).type(gpu_dtype)

x_gpu = torch.randn(64, 3, 32, 32).type(gpu_dtype)
x_var_gpu = Variable(x.type(gpu_dtype)) # Construct a PyTorch Variable out of your input data
ans = fixed_model_gpu(x_var_gpu)        # Feed it through the model! 

# Check to make sure what comes out of your model
# is the right dimensionality... this should be True
# if you've done everything correctly
np.array_equal(np.array(ans.size()), np.array([64, 10]))

#Train the model.
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(fixed_model_gpu.parameters(),lr=0.001)

def train(model, loss_fn, optimizer, num_epochs = 1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype).long())

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
torch.cuda.random.manual_seed(12345)
fixed_model_gpu.apply(reset)
train(fixed_model_gpu, loss_fn, optimizer, num_epochs=1)
check_accuracy(fixed_model_gpu, loader_val)


#Train a great model on CIFAR-10!
model = = nn.Sequential(
                   nn.Conv2d(3, 32, kernel_size=3,padding = 1),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm2d(32)
                   nn.Conv2d(32, 64, kernel_size=3,padding = 1),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm2d(64),
                   nn.MaxPool2d(2,stride = 2),
                   Flatten(), # see above for explanation
                   nn.Linear(16384,1024),# 16384=64*32*32 input size
                   nn.ReLU(inplace=True),
                   nn.BatchNorm1d(num_features=1024),
                   nn.Linear(1024, 10), # affine layer
                    )
model_gpu = model.type(gpu_dtype)
loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
optimizer = optim.RMSprop(model_gpu.parameters(),lr=0.001)
total_epochs = 10

train(model_gpu, loss_fn, optimizer, num_epochs=total_epochs)
check_accuracy(model_gpu, loader_val)

best_model = model_gpu
check_accuracy(best_model, loader_test)
