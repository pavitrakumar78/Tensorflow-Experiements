# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 21:09:54 2017

@author: Pavitrakumar
"""
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py
#https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-6-mnist_nn_batchnorm.ipynb

import tensorflow as tf
from keras.datasets import mnist
import numpy as np

tf.reset_default_graph()
sess = tf.InteractiveSession()


def construct_network(inp_shape, out_shape, hidden_dims = [10,20], activation = tf.nn.relu):
    #Simple Multi-layer-NN - Only dense layers - no conv
    with tf.variable_scope("nnet"): #not really necessary - for clarity here. Maybe useful in specific cases where we reuse vars
        X = tf.placeholder(shape = [None, inp_shape], dtype = tf.float32) #don't go lower than float32; else will get 'nan' in weights
        Y = tf.placeholder(shape = [None, out_shape], dtype = tf.float32)
        
        net = X
        for index,layer_size in enumerate(hidden_dims):
            with tf.variable_scope('layer{}'.format(index)): #not really necessary - for clarity here. Maybe useful in specific cases where we reuse vars
                net = tf.layers.dense(inputs = net, units = layer_size, activation = activation)
                net = tf.layers.batch_normalization(net) #improves acc? - possibly.
        
        logits = tf.layers.dense(inputs = net, units = out_shape)
        
        #entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
        #loss = tf.reduce_mean(entropy) 
        
        #does the same as above?...pretty much.
        loss = tf.losses.softmax_cross_entropy(logits = logits, onehot_labels= Y)
            
        #to train network
        train_expr = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        
        #we need other metrics such as accuracy and loss - we already have loss (see above)
        #accuracy:
        preds = tf.nn.softmax(logits)
        pred_class = tf.argmax(preds, 1)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        
    return X, Y, train_expr, loss, pred_class, accuracy, logits


#Load data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#rehspaing 3d into 2d
x_train = np.reshape(x_train, newshape = (x_train.shape[0], np.prod(x_train.shape[1:])))
x_train = (x_train - 127.5 )/ 127.5
x_test = np.reshape(x_test, newshape = (x_test.shape[0], np.prod(x_test.shape[1:])))
x_test = (x_test - 127.5 )/ 127.5
#one-hot encoding
temp = np.zeros((y_train.shape[0], y_train.max()+1))
temp[np.arange(y_train.shape[0]),y_train] = 1
y_train = temp
temp = np.zeros((y_test.shape[0], y_test.max()+1))
temp[np.arange(y_test.shape[0]),y_test] = 1
y_test = temp

#better to return this as a dict. look:Project-DeepReinforcementLearning/Atari/MAP-DQN run.py and Net_A3C.py
X, Y, train_expr, loss, pred_class, accuracy, logits = construct_network(inp_shape = x_train.shape[1], 
                                                           out_shape = y_train.shape[1],
                                                           hidden_dims = [32,32])


epochs = 20
batch_size = 128
num_steps = int(x_train.shape[0]/batch_size)


sess.run(tf.global_variables_initializer())

#training
for epoch in range(epochs):
    for i in range(num_steps):
        sample_indices = np.random.choice(x_train.shape[0], batch_size)
        x_sample = x_train[sample_indices]
        y_sample = y_train[sample_indices]
        acc, _loss, _logits, _ = sess.run(fetches = [accuracy, loss, logits, train_expr],feed_dict = {X: x_sample, Y: y_sample})
    print("Epochs: ",epoch," Training Loss: ",_loss, "Training Accuracy:", acc)


#testing
test_acc = sess.run(fetches = [accuracy],feed_dict = {X: x_test, Y: y_test})
print("Testing Accuracy: ",test_acc)
