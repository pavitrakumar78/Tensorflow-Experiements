# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:12:28 2017

@author: Pavitrakumar
"""

import numpy as np
import tensorflow as tf
from keras.datasets import mnist



#Simple Logistic regression 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#rehspaing 3d into 2d
x_train = np.reshape(x_train, newshape = (x_train.shape[0], np.prod(x_train.shape[1:])))

x_test = np.reshape(x_test, newshape = (x_test.shape[0], np.prod(x_test.shape[1:])))


#one-hot encoding
temp = np.zeros((y_train.shape[0], y_train.max()+1))
temp[np.arange(y_train.shape[0]),y_train] = 1
y_train = temp
temp = np.zeros((y_test.shape[0], y_test.max()+1))
temp[np.arange(y_test.shape[0]),y_test] = 1
y_test = temp

#define placeholders (generalized shape)
X = tf.placeholder(tf.float32, shape = [None,x_train.shape[1]])
Y = tf.placeholder(tf.float32, [None, y_train.shape[1]])

#init weights and biases (fixed shape)
#dim(X.W) = (ROWS,784)x(784,10) = (ROWS,10) (Y)

X_features_dim = X.shape[1]
Y_num_classes = Y.shape[1]

W = tf.Variable(tf.random_normal(shape=[X_features_dim.value, Y_num_classes.value]), name="weights")
b = tf.Variable(tf.random_normal(shape=[1, Y_num_classes.value]), name="biases")


#hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost/loss function
#loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

#Way2
logits = tf.matmul(X, W) + b
entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y)
#entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
# cost/loss function
loss = tf.reduce_mean(entropy) 

#https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
#http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/

#try adagrad,adam,rms prop etc.,
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


preds = tf.nn.softmax(logits)
preds_class = tf.argmax(preds, 1)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

#whatever we want to extract/want to know/want to use for other calculations, put them as an expression!
#now, if that expression evaluates to a float, to know the result after computing the initial graph, use expr.eval(dict)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        for i in range(int(x_train.shape[0]/100)):
            sample_indices = np.random.choice(x_train.shape[0], 100)
            x_sample = x_train[sample_indices]
            y_sample = y_train[sample_indices]
            _logits, _loss, _ = sess.run(fetches = [ logits, loss, train],feed_dict = {X: x_sample, Y: y_sample})
        print("Epoch: ",epoch," Training Loss: ",_loss)
        #print("Training Accuracy: ",accuracy)
        #calculating accuracy
    
    #testing:
    #getting only preds
    only_preds = preds_class.eval(feed_dict = {X: x_test})
    #getting only acc
    acc = accuracy.eval(feed_dict = {X: x_test, Y: y_test}, session = sess)
    print("Testing Accuracy:",acc)
    #Manual checking
    print(np.sum(only_preds == np.argmax(y_test,axis=1))/y_test.shape[0])
    
#http://web.stanford.edu/class/cs20si/lectures/notes_03.pdf
#https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-4-mnist_introduction.py