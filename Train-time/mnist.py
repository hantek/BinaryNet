
from __future__ import print_function

import sys
import os
import time
from collections import OrderedDict
import cPickle as pickle
import gzip
import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T
import lasagne

# if os.environ['HOME'] == '/home/hantek':
#     os.environ['PYLEARN2_DATA_PATH'] = '/home/hantek/projects/OnchipBNN'
#     from pylearn2.datasets.mnist import MNIST10 as MNIST
# else:
#     from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

import binary_net

import pdb
#theano.config.compute_test_value = 'warn'  # 'off' # Use 'warn' to activate this feature


def quantization (array,num_bits):# DAC Quantization
    # This quantization will limit input array in range [0, 1)
   
    max_num = 1.0 - 0.5**num_bits 
    min_num = 0
    num_levels = (2. ** num_bits)
    
    array_res = np.empty_like(array)
    array_res[:] = array  
    array_res = array_res.reshape(array_res.size,)

    levels = np.linspace(min_num,max_num,num_levels)
    cnt = 0    
    for i in np.nditer(array):
        tmp = np.abs(levels - i)
        index = (tmp == np.min(np.abs(levels - i)))        
        array_res[cnt] = levels[index]
        cnt = cnt + 1        
    return array_res.reshape((array.shape))


if __name__ == "__main__":
    
    # BN parameters
    batch_size = 100
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 4096  # 96
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 100
    print("num_epochs = "+str(num_epochs))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    LR_start = .003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "mnist_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading MNIST dataset...')
    
    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = False)
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = False)
    test_set = MNIST(which_set= 'test', center = False)
    
    # bc01 format    
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_set.X = 2* train_set.X.reshape(-1, 1, 28, 28) - 1.
    valid_set.X = 2* valid_set.X.reshape(-1, 1, 28, 28) - 1.
    test_set.X = 2* test_set.X.reshape(-1, 1, 28, 28) - 1.
    
    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])    
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])
    
    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    layer_inputs = []
    layer_outputs = []
    layer_testoutputs = []

    layer_faketargets = []
    layer_fakelosses = []
    layer_updates = []
    layer_deltas = []

    layer_fwdfns = []
    layer_bcwdfns = []
    layer_testfwdfns = []

    for k in range(n_hidden_layers):
        if k == 0:
            input = T.tensor4('original_4d_inputs')
            layer_inputs.append(input)
            l_input = lasagne.layers.InputLayer(
                    shape=(None, 1, 28, 28),
                    input_var=layer_inputs[-1])
        else:
            layer_inputs.append(T.matrix(str('k') + '\'s input'))
            l_input = lasagne.layers.InputLayer(
                    shape=(None, num_units),
                    input_var=layer_inputs[-1])

        mlp = binary_net.DenseLayer(
                l_input, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  

        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)
        
        kth_output = lasagne.layers.get_output(mlp, deterministic=False)
        layer_outputs.append(kth_output)

        layer_faketargets.append(T.matrix(str('k') + '\'s_layer_target'))
        fakeloss = T.sum(kth_output * layer_faketargets[-1])
        layer_fakelosses.append(fakeloss)

        layer_deltas.append(T.grad(fakeloss, wrt=layer_inputs[-1]))

        if binary:
            # W updates
            W = lasagne.layers.get_all_params(mlp, binary=True)

            print("layer %d: " % k),
            print(W)

            W_grads = binary_net.compute_grads(fakeloss, mlp)
            updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
            updates = binary_net.clipping_scaling(updates, mlp)

            # other parameters updates
            params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
            updates = OrderedDict(
                    updates.items() + \
                    lasagne.updates.adam(loss_or_grads=fakeloss, params=params, learning_rate=LR).items())
        else:
            params = lasagne.layers.get_all_params(mlp, trainable=True)
            updates = lasagne.updates.adam(loss_or_grads=fakeloss, params=params, learning_rate=LR)

        layer_updates.append(updates)

        fwdfn = theano.function([layer_inputs[-1]], layer_outputs[-1])
        layer_fwdfns.append(fwdfn)

        bcwdfn = theano.function(
                [layer_inputs[-1], layer_faketargets[-1], LR],
                layer_deltas[-1], updates=updates)
        layer_bcwdfns.append(bcwdfn)
        
        ## stuffs for test
        layer_testoutputs.append(lasagne.layers.get_output(mlp, deterministic=True))
        testfwdfn = theano.function([layer_inputs[-1]], layer_testoutputs[-1])
        layer_testfwdfns.append(testfwdfn)

    classifier_input = T.matrix('classifier_input')

    l_classifier_input = lasagne.layers.InputLayer(
                shape=(None, num_units),
                input_var=classifier_input)

    mlp = binary_net.DenseLayer(
                l_classifier_input, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)    
    
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    classifier_delta = T.grad(loss, wrt=classifier_input)

    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)

        print("classfifer: "),
        print(W)

        W_grads = binary_net.compute_grads(loss, mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    classifier_train_fn = theano.function(
        [classifier_input, target, LR], [loss, classifier_delta], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1),
                            T.argmax(target, axis=1)),
                      dtype=theano.config.floatX)
    classifier_val_fn = theano.function([classifier_input, target], [test_loss, test_err])

    print('Training...')
    binary_net.train(
        layer_fwdfns, layer_testfwdfns,
        classifier_train_fn, classifier_val_fn,
        layer_bcwdfns, 
        n_hidden_layers,
        mlp,
        batch_size,
        LR_start,LR_decay,
        num_epochs,
        train_set.X,train_set.y,
        valid_set.X,valid_set.y,
        test_set.X,test_set.y,
        save_path,
        shuffle_parts)
