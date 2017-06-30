# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:07:11 2017

@author: Pavitrakumar
"""

"""
Simple TensorFlow exercises
You should thoroughly test your code
https://github.com/chiphuyen/tf-stanford-tutorials/tree/master/assignments/exercises
"""

import tensorflow as tf

sess = tf.Session()


###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))

print(sess.run(out,feed_dict = {x:4,y:1}))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x = tf.random_uniform([],minval = -1,maxval = 1, dtype = tf.float32)  # Empty array as shape creates a scalar.
y = tf.random_uniform([],minval = -1,maxval = 1, dtype = tf.float32)
out = tf.case({tf.less(x,y):lambda:tf.add(x,y),tf.less(y,x):lambda:tf.subtract(x,y)},default = lambda: tf.constant(0.0),exclusive=True)

print(sess.run(out,feed_dict = {x:5,y:1}))

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.fill(dims = x.shape, value = 0)
out = tf.equal(x,y)

print(sess.run(out,feed_dict = {x:[[1,2,3],[3,2,1]],y:[[1,2,3],[3,2,1]]}))

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
                 30.97266006,  26.67541885,  38.08450317,  20.74983215,
                 34.94445419,  34.45999146,  29.06485367,  36.01657104,
                 27.88236427,  20.56035233,  30.20379066,  29.51215172,
                 33.71149445,  28.59134293,  36.05556488,  28.66994858])

#cond_satisfy = x>30 # also valid
cond_satisfy = tf.where(x>30) #returns something like [T,F,T,T,T,F...]
out = tf.gather(x, cond_satisfy) #returns the values of x where cond_satisfy is true

print(sess.run(out))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

out = tf.diag(tf.range(1,limit=7))

print(sess.run(out))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

x = tf.random_uniform(shape = [10,10], minval = 1, maxval = 10)
#x = tf.round(x)
out = tf.matrix_determinant(x)

print(sess.run(out))

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
out = tf.unique(x)[0] #returns ([unique ele],[uinque ele indices])
print(sess.run(out))

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

x = tf.random_normal(shape = [300,300])
y = tf.random_normal(shape = [300,300])

a = tf.reduce_mean(tf.squared_difference(x, y)) #mean squared error
b = tf.reduce_sum(tf.abs(x-y))

out = tf.cond(tf.less(tf.reduce_mean(x-y),0),lambda: a, lambda: b)

print(sess.run(out))

###############################################################################

x = tf.fill(dims = [100,100,3,32], value = 1.0)
y = tf.fill(dims = [32],value = 1.0)
opt = tf.add(x,y)


print(sess.run(opt))
