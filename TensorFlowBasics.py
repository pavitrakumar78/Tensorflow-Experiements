# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:05:15 2017

@author: Pavitrakumar
"""

import tensorflow as tf

tf.__version__

#GRAPHS GRAPHS GRAPHS!
#Everything is computed using graphs

#Add 2 numbers - 3 ways

x1 = 1
y1 = 2
result1 = tf.add(x1,y1)
#<tf.Tensor 'Add:0' shape=() dtype=int32>
#Nothing is calculated outside of a session

x2 = tf.constant(3.0, tf.float32)
y2 = tf.constant(4.0) # also tf.float32 implicitly
#<tf.Tensor 'Const:0' shape=() dtype=float32>
result2 = tf.add(x2, y2)

x3 = tf.placeholder(tf.float32)
y3 = tf.placeholder(tf.float32)
result3 = x3 + y3  # + provides a shortcut for tf.add(a, b)


with tf.Session() as sess:
    print(sess.run(result1)) #3
    print(sess.run(result2)) #7.0
    #print(sess.run(result3)) #NO! we need to input data ourselves now!
    print(sess.run(result3,feed_dict = {x3 : 1, y3 : 2}))
    #print(sess.run(result3,feed_dict = {x3 : tf.constant(10), y3 : 2})) #cannot feed tensors
    #TypeError: The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars, strings, lists, or numpy ndarrays.

    #Vectors and Matrices
    #Dim = Rank; 1d = rank1 ; 2d = rank2 ; 3d = rank3
    
    # constant of 1d tensor (vector)
    a = tf.constant([2, 2], name="v1")
    print(sess.run(a))
    # constant of 2x2 tensor (matrix)
    b = tf.constant([[0, 1], [2, 3]], name="b")
    print(sess.run(b))
    c = tf.zeros([2, 3], tf.int32) 
    #[[0, 0, 0], [0, 0, 0]]
    print(sess.run(c))
    d = tf.fill(dims = [2,3],value = 0.5, name = None) 
    #[[0.5,0.5,0.5],[0.5,0.5,0.5]]
    print(sess.run(d))
    #unlike numpy, we  specify shape/dim like -> [rows,cols]
    e = tf.constant(1,shape=[3,4])
    print(sess.run(e))
    
    #tf.int16 == np.int16 #same!

#variables vs constants
#1. A constant is constant. A variable can be assigned to, its value can be changed.
#2. A constant's value is stored in the graph and its value is replicated wherever the graph is
#   loaded. A variable is stored separately, and may live on a parameter server.

#create variable a with scalar value
a = tf.Variable(2, name="scalar")
#create variable b as a vector
b = tf.Variable([2, 3], name="vector")
assign_op = b.assign([1,2]) #dim must be same!
#You have to initialize variables before using them. 
W = tf.Variable(tf.truncated_normal([700, 10]))

init = tf.global_variables_initializer() #comes after init vars like above

with tf.Session() as sess:
    #print(a.eval()) #ERROR: Attempting to use uninitialized value scalar_4 
    #Way1
    sess.run(assign_op)
    print(b.eval())
    #Way2
    sess.run(init)
    print(W)
    print(W.eval())


