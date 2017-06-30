# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:45:52 2017

@author: Pavitrakumar
"""

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt


sess = tf.InteractiveSession()


#conv2d basic I/O
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image = x_train[0].reshape((1,28,28,1))
image = image.astype(np.float32) #safe!
#image.dtype = np.float32
print(image.shape)
plt.imshow(image.reshape((28,28)), cmap='Greys')

#https://stackoverflow.com/questions/12362094/why-setting-dtype-on-numpy-array-changes-dimensions
#data_format for tf in conv is by default: NHWC - No. of samples, height, width, channels


#################### 2D IMAGE - opt: 32 conv2d filters with size 5x5 with padding: VALID ########################
#init weights of dim:
#filter_height, filter_width, in_channels, out_channels   [OR]
#filter_size, filter_size, input_channels, output_channels(or num. filters)
filter_weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32],dtype=tf.float32)) #inp. channel is only 1
conv2d = tf.nn.conv2d(image, filter = filter_weights, strides=[1, 1, 1, 1], padding='VALID')
#NOTE:: FLOAT64 DOES NOT WORK!
conv2d_img = conv2d.eval()
print(conv2d_img.shape) #(1, 24, 24, 32)
"""
How to get (1,24,24,32) using formula:
32 - filter size (we defined it)
(1,24,24) 
((W-F+(2*P))/S)+1 
W - size of input: 28,28
F - size of filter: 5
P - padding - we use valid padding (NO ZERO PADDING) = 0
S - stride size - (1,1) = 1
((28-5+2*0)/1)+1 = (23/1)+1 = 24
"""
#The stride of the sliding window for each dimension of input.  [num samples, height, width, channels]
conv2d = tf.nn.conv2d(image, filter = filter_weights, strides=[1, 2, 2, 1], padding='VALID')
conv2d_img = conv2d.eval()
print(conv2d_img.shape) #(1, 12, 12, 32)

#################### 2D and 3D IMAGE - opt: 32 conv2d filters with size 5x5 with padding: VALID ########################

#Note: Dont confuse dimension req. for filter weights and dimension req. for strides
#filter weights = filter_size, filter_size, input_channels, output_channels(or num. filters)
#strides = samples, height, width, channels (same as input (image) dim)

#image = np.random.uniform(size = (1,28,28,3))
n_dim = 3 # 3 for 3d and 1 for 2d
image = np.random.randint(1,10,size = (1,5,5,n_dim))

image = image.astype(np.float32) #safe!

print(image.shape)
#plt.imshow(image[0])

filter_weights = tf.Variable(tf.round(tf.random_uniform(shape=[2, 2, n_dim, 32], minval=0, maxval=10, dtype=tf.float32))) #inp. channel is 3
conv2d = tf.nn.conv2d(image, filter = filter_weights, strides=[1, 2, 2, 1], padding='VALID')

sess.run(tf.global_variables_initializer())

conv2d_img = conv2d.eval()
print(conv2d_img.shape)
"""
image : (2D)
      [[ 9.,  9.,  8.,  7.,  8.],
       [ 5.,  5.,  7.,  8.,  5.],
       [ 4.,  8.,  9.,  6.,  8.],
       [ 4.,  5.,  3.,  4.,  8.],
       [ 1.,  1.,  3.,  7.,  1.]] 
filter: (0th)
      [[ 1.,  2.],
       [ 3.,  4.]] 
output: (0th)
      [[ 62.,  75.],
       [ 52.,  46.]] 
calculations:
     |9, 9| x |1, 2| = sum(|9, 18 |  =  9 + 18 + 15 + 20 = 62 = (0,0)th ele of output
     |5, 5|   |3, 4|       |15, 20|) 
     {{stride = 2x2}} meaning we move 2 across untill end (horizontal) and 2 vertical after we finish a row
     after stride:
     |8, 7| x |1, 2| = sum(|8, 14 |  = 8 + 14 + 21 + 32 = 75 = (0,1)th ele of output
     |7, 8|   |3, 4|       |21, 32|) 
     we can no-long do stride because we are using VALID padding (i.e we cannot zero pad at the end of row 1 and row 2
     and do another conv op. in the same row, so we move vertically now)
     after stride:
     |4, 8| x |1, 2| = sum(|4, 16 |  = 4 + 16 + 12 + 20 = 52 = (1,0)th ele of output
     |4, 5|   |3, 4|       |12, 20|)
     now we continue horizontally 
     |9, 6| x |1, 2| = sum(|9, 12|   = 9 + 12 + 9 + 16 = 46 = (1,1)th ele of output
     |3, 4|   |3, 4|       |9, 16|)
"""

"""
image: (3D)
      [[[ 2.,  3.,  2.,  7.,  3.],       [[[ 2.,  8.,  8.,  4.,  7.],        [[[ 9.,  3.,  9.,  4.,  7.],
        [ 9.,  2.,  8.,  4.,  7.],         [ 8.,  6.,  7.,  8.,  6.],          [ 7.,  7.,  9.,  4.,  8.],
        [ 3.,  4.,  2.,  9.,  2.],         [ 4.,  2.,  1.,  9.,  8.],          [ 1.,  4.,  8.,  9.,  7.],
        [ 3.,  6.,  4.,  6.,  2.],         [ 9.,  3.,  6.,  9.,  5.],          [ 8.,  5.,  4.,  8.,  1.],
        [ 7.,  7.,  6.,  5.,  7.]]]        [ 7.,  6.,  9.,  8.,  2.]]]         [ 9.,  5.,  4.,  6.,  6.]]]
filter: (0th)
      [[ 4.,  3.],     [[ 6.,  4.],       [[ 2.,  0.],
       [ 1.,  1.]]      [ 2.,  1.]]        [ 4.,  4.]]
output: (0th) (same overall shape as previous case of 2d) (1, 2, 2, 32)
     [[ 168.,  197.],
      [ 140.,  172.]],
calculations:
     |2, 3| x |4, 3| = sum(|8, 9|  =  8 + 9 + 9 + 2 = 28  (filter:0  dim:0)
     |9, 2|   |1, 1|       |9, 2|)
     
     |2, 8| x |6, 4| = sum(|12, 32|  =  12 + 32 + 16 + 6 = 66  (filter:0  dim:1)
     |8, 6|   |2, 1|       |16, 6 |)
     
     |9, 3| x |2, 0| = sum(|18, 0 |  =  18 + 0 + 28 + 28 = 74  (filter:0  dim:2)
     |7, 7|   |4, 4|       |28, 28|)
     
     28 + 66 + 74 = 168 = we do sum of sums for 3D - this is why the dims are same for 2D and 3D 
     i.e another way of looking at it is, since the image is of size (h, w, 3) and kernel is (k_h, k_w, 3),
     we only need to convolve along the first 3 dimentions since the 3rd dim is the same - no need to conv along that dim.
     (rest of the steps are similar to 2D conv above)
     #https://www.quora.com/Why-do-we-use-2D-convolution-for-RGB-images-which-are-in-fact-3D-data-cubes-Shouldnt-we-use-3D-convolution-for-RGB-and-4D-for-video#!n=12
"""


#################### MAX-pooling  ########################

n_dim = 1

image = np.random.randint(1,10,size = (1,5,5,n_dim))

image = image.astype(np.float32) #safe!

print(image.shape)

#no weights for pool, all we are doing is summing the given input in small windows
#values/meanings of of ksize dim = same as strides dim = same as input dim
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
pool_img = pool.eval()
print(pool_img.shape) #(1, 4, 4, n_dim)


pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
pool_img = pool.eval()
print(pool_img.shape) #(1, 5, 5, n_dim)






