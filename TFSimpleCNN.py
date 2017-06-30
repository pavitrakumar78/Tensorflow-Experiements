# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:05:22 2017

@author: Pavitrakumar
"""

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time
tf.set_random_seed(123)

#https://www.tensorflow.org/tutorials/layers
#https://github.com/carpedm20/DCGAN-tensorflow/blob/66884d3dbbde4033ffe5305bb0053fc4625d0624/ops.py
#https://github.com/m516825/Conditional-GAN/blob/master/model.py
#https://github.com/chiphuyen/tf-stanford-tutorials/blob/master/examples/07_convnet_mnist.py
#https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-1-mnist_cnn.py
#http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
#http://ruishu.io/2016/12/27/batchnorm/

def apply_activation(inp, activation = "relu"):
    if activation == 'relu' : return tf.nn.relu(inp)
    elif activation == 'elu' : return tf.nn.elu(inp)
    elif activation == 'tanh' : return tf.nn.tanh(inp)
    elif activation == 'sigmoid' : return tf.nn.sigmoid(inp)
    elif activation == 'softmax' : return tf.nn.softmax(inp)
    elif activation == 'softplus' : return tf.nn.softplus(inp)


def conv_layer(inp, num_inp_channels, num_filters, kernel_size, stride_dim, name = "conv2d", padding = "VALID", add_bias = True, stddev=0.02,):
    with tf.variable_scope(name):
        kernel_size = [kernel_size]
        stride_dim = [stride_dim]
        kernel_size = list(kernel_size)
        if len(kernel_size) == 1: #[5] assumed as -> [5,5]
            filter_weights = tf.get_variable(name = 'conv_filter_weights', shape = [kernel_size[0], kernel_size[0], num_inp_channels, num_filters], initializer=tf.truncated_normal_initializer())
        else: #[5,5] assumed as -> [5,5] (same applies for strides)
            filter_weights = tf.get_variable(name = 'conv_filter_weights', shape = [kernel_size[0], kernel_size[1], num_inp_channels, num_filters], initializer=tf.truncated_normal_initializer())
        stride_dim = list(stride_dim)
        if len(stride_dim) == 2:
            c = tf.nn.conv2d(input = inp, filter = filter_weights, strides = [1,stride_dim[0],stride_dim[1],1], padding = padding, data_format = "NHWC", name = 'conv_layer')
        else:
            c = tf.nn.conv2d(input = inp, filter = filter_weights, strides = [1,stride_dim[0],stride_dim[0],1], padding = padding, data_format = "NHWC", name = 'conv_layer')
        
        if add_bias:
            biases = tf.get_variable(name = 'conv_bias', shape = [num_filters], initializer=tf.random_normal_initializer())
            c = tf.add(c,biases)
        
        return c
"""
def pool_layer(inp, kernel_size, stride_dim, name = "pool", padding = "VALID"):
    with tf.variable_scope(name):
        p = tf.nn.max_pool(inp, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_dim, stride_dim, 1], padding = padding, name = "pool_layer")
        return p

def fc_layer(inp, num_hidden_units, name = "fc", stddev=0.02):
    input_shape = inp.get_shape().as_list()
    with tf.variable_scope(name):
        #dim = input_shape[1]
        reshape_dim = np.prod(input_shape[1:]) #input features
        weight = tf.get_variable("fc_weight", [reshape_dim, num_hidden_units], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("fc_bias", [num_hidden_units], initializer=tf.truncated_normal_initializer())
        #read about "-1" https://www.tensorflow.org/api_docs/python/tf/reshape
        #-1 can be used to infer shape automatically
        #https://github.com/Kyubyong/tensorflow-exercises/blob/master/Tensor_Transformations_Solutions.ipynb
        inp = tf.reshape(inp, [-1, reshape_dim])
        #above means that [row,col]-> input_features -> given constant so -1 automatically infers the rest of the shape
        opt = tf.add(tf.matmul(inp,weight), bias, name='fc_result')
        
        return opt
"""

def conv_layer_simple(inp, num_filters, kernel_size, stride_dim, name = "conv2d", padding = "valid", add_bias = False, stddev=0.02,):
    with tf.variable_scope(name):
        #https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
        c = tf.layers.conv2d(inputs = inp, filters = num_filters, kernel_size = kernel_size, strides = stride_dim, padding = padding, data_format = "channels_last", name = 'conv_layer')
        return c

def pool_layer(inp, kernel_size, stride_dim, name = "pool", padding = "valid"):
    with tf.variable_scope(name):
        p = tf.layers.max_pooling2d(inp, kernel_size, stride_dim, padding = padding)
        return p

def dense_layer(inp, num_hidden_units, name = "dense"):
    with tf.variable_scope(name):
        d = tf.layers.dense(inputs=inp, units=num_hidden_units)
        return d

def flatten_layer(inp):
    input_shape = inp.get_shape().as_list()
    reshape_dim = np.prod(input_shape[1:]) #input features
    flat = tf.reshape(inp, [-1, reshape_dim])
    return flat
    #return tf.contrib.layers.flatten(inp) #trying not to rely on contrib that much.

def batchnorm_layer(inp):
    #confusion!
    #tf.nn.batch_normalization or tf.layers.batch_normalization or tf.contrib.layers.batch_norm (???)
    #https://github.com/tensorflow/tensorflow/issues/7091
    #http://ruishu.io/2016/12/27/batchnorm/
    #probably need to use with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    bn_fn = tf.nn.batch_normalization       #https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization  #-- not used as often as below 2 
        
    bn_fn = tf.contrib.layers.batch_norm    #https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    bn_fn = tf.layers.batch_normalization   #https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization

    bn = bn_fn(inp)
    
    return bn
    
sess = tf.InteractiveSession()


(x_train, y_train), (x_test, y_test) = mnist.load_data()
image = x_train[0].reshape((1,28,28,1))
image = image.astype(np.float32) 
print(image.shape)
plt.imshow(image.reshape((28,28)), cmap='Greys')


#constructing network
x_train = x_train[...,np.newaxis] #(60000, 28, 28, 1)
temp = np.zeros((y_train.shape[0], y_train.max()+1))
temp[np.arange(y_train.shape[0]),y_train] = 1
y_train = temp

X = tf.placeholder(tf.float32, [None, x_train.shape[1],x_train.shape[2],x_train.shape[3]], name="X_placeholder")
Y = tf.placeholder(tf.float32, [None, y_train.shape[1]], name="Y_placeholder")

#images_input = tf.reshape(X, shape=[1, 28, 28, 1]) 

#conv_net = conv_layer_simple(X,32,5,1,name="conv_l1")
conv_net = conv_layer(X,x_train.shape[-1],32,5,1,name="conv_l1")
conv_net = apply_activation(conv_net,activation = "relu")
conv_net = pool_layer(conv_net,2,2,name = "pool1")
#We don't acually need names for pooling layers - no matrix involved other than a non-learnable filter
#remove it later
#conv_net = conv_layer_simple(conv_net,64,5,1,name="conv_l2")
conv_net = conv_layer(conv_net,32,64,5,1,name="conv_l2")
conv_net = apply_activation(conv_net,activation = "relu")
conv_net = pool_layer(conv_net,2,2,name = "pool2")

conv_net = flatten_layer(conv_net) #same as above!

conv_net = dense_layer(conv_net,512,name="dense1")
conv_net = apply_activation(conv_net, activation = "relu")
conv_net = dense_layer(conv_net,10,name="dense2") #linear activation (default)


entropy = tf.nn.softmax_cross_entropy_with_logits(logits = conv_net,labels = Y)
loss = tf.reduce_mean(entropy, name='loss')

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

sess.run(tf.global_variables_initializer())

time_start = time.time()
for step in range(200):
    sample_indices = np.random.choice(x_train.shape[0], 100)
    x_sample = x_train[sample_indices]
    y_sample = y_train[sample_indices]
    _logits, _loss, _ = sess.run(fetches = [conv_net, loss, optimizer],feed_dict = {X: x_sample, Y: y_sample})
    
    classes = apply_activation(_logits,activation = "softmax")
    correct_preds = tf.equal(tf.argmax(classes, 1), tf.argmax(y_sample, 1)).eval()
    acc = np.sum(correct_preds)/len(correct_preds)
    
    print("Step: ",step," Training Loss: ",_loss, "Training Accuracy:", acc)

time_end = time.time()
time_taken = time_end - time_start
print("Total time taken: ",time_taken," secs")
             
#conv_layer(...) -> takes 28 secs for 200 steps (~95% training acc)
#conv_layer_simple(...) -> takes 63 secs for 200 steps (~98-100% training acc)

#also check for tf.nn.pooling and tf.layers.pooling