# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:26:04 2017

@author: Pavitrakumar
"""

import tensorflow as tf
import os
from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from six.moves import urllib
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
import copy
tf.set_random_seed(123)
import time

#https://github.com/chiphuyen/tf-stanford-tutorials/blob/master/assignments/style_transfer/vgg_model.py

def download(download_link, file_name):
    if os.path.exists(file_name):
        print("Dataset ready")
        return
    print("Downloading the VGG pre-trained model")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)


def _weights(vgg_layers, layer, expected_layer_name):
    #Return the weights and biases already trained by VGG
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W, b.reshape(b.size)

def _conv2d_relu(vgg_layers, prev_layer, layer, layer_name):
    with tf.variable_scope(layer_name):
        vgg_w , vgg_b = _weights(vgg_layers, layer, layer_name)
        c = tf.nn.conv2d(input = prev_layer, filter = vgg_w, strides = [1,1,1,1], padding = "SAME", data_format = "NHWC", name = 'conv_layer')
        c = tf.add(c,vgg_b)
        return tf.nn.relu(c)

def _avgpool(prev_layer):
    #p = tf.layers.average_pooling2d(prev_layer,pool_size=2,strides=2,padding='valid')
    p = tf.nn.avg_pool(prev_layer,[1,2,2,1],[1,2,2,1],padding='SAME')
    return p

def load_vgg(path, image_placeholder):
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']

    graph = {} 
    graph['conv1_1']  = _conv2d_relu(vgg_layers, image_placeholder, 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(vgg_layers, graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(vgg_layers, graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(vgg_layers, graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(vgg_layers, graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(vgg_layers, graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(vgg_layers, graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(vgg_layers, graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(vgg_layers, graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(vgg_layers, graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(vgg_layers, graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(vgg_layers, graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(vgg_layers, graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(vgg_layers, graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(vgg_layers, graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(vgg_layers, graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph

#need to minimize 2 losses - content loss and style loss
#http://web.stanford.edu/class/cs20si/assignments/a2.pdf

def get_content_loss(f, p):
    #F is the feature representation of the generated image
    #P is the feature representation of the content image in layer 'conv4_2​'
    #const term is 1/(4s) in which s is the product of the dimension of P.
    
    #const_term = 4*np.prod(p.shape[1:])
    const_term = 4*p.size #also works!
    content_loss = (1/const_term)*(tf.reduce_sum(tf.squared_difference(f,p)))
    
    #print("csloss",content_loss.eval())
    return content_loss

def get_layer_style_loss(g, a):
    #N is the third dimension of the feature map [w,x,y,[z]]
    #M is the product of the first two dimensions of the feature map [w,[x,y],z]
    #in above dims, we exlcude the first dim because in conv, 1st dim = no. of samples.
    #a is the feature rep. of the original image
    #g is the feature rep. of the generated image
    #n,m - which feature map?? input image or the generated image?
    
    n = a.shape[3]  #a or g? - both feature maps of same dim! - num filters
    #m = height x width
    m = int(np.prod(a.shape[1:3]))      #gives INF as o/p in final loss calculation -- this is np.int32 type -- 
    #but encapsulating above with int(...) works!
    #m = a.shape[1] * a.shape[2]     #gives correct o/p   -- this is normal python int type
    
    #probably because individual values to dims to reshape must be of same type
    """
    #Dont mix types! (numpy vs python int to -> tf)
    """
    A = tf.reshape(a, (m,n))
    G = tf.reshape(g, (m,n))
    
    A = get_gram_matrix(A)
    G = get_gram_matrix(G)
    
    """
    #Dont do stuff like this - mixing up ints and floats - will get nan's and inf's
    #constant_term = 1/(4*tf.square(n)*tf.square(m)) #by default it is float64
    #constant_term = tf.cast(constant_term, tf.float32)
    #layer_style_loss = tf.multiply(constant_term,tf.reduce_sum(tf.squared_difference(g,a)))
    """
    layer_style_loss = tf.reduce_sum(tf.squared_difference(G,A))/(4*n*n*m*m)
    ## try (G - A) vs (g - a) - different losses!?
    
    return layer_style_loss

def get_gram_matrix(mat):
    return tf.matmul(tf.matrix_transpose(mat),mat)


def get_style_loss(a,vgg_model, style_layers):
    
    E = []
    for index,layer in enumerate(style_layers):
        E_i = get_layer_style_loss(vgg_model[layer],a[index])
        E.append(E_i)
    E_weights = [0.5, 1.0, 1.5, 3.0, 4.0]
    style_loss= tf.reduce_sum(tf.multiply(E, E_weights))
    
    return style_loss


def get_combined_loss(sess, image_placeholder, content_image, style_image, model, style_layers, content_layer):
    with tf.variable_scope('loss'):
        sess.run(image_placeholder.assign(content_image)) # assign content image to the input variable
        p = sess.run(model[content_layer])
        content_loss = get_content_loss(model[content_layer], p)

        sess.run(image_placeholder.assign(style_image))
        A = sess.run([model[layer_name] for layer_name in style_layers])                              
        style_loss = get_style_loss(A, model, style_layers)    
        
        content_loss_weight = 1
        style_loss_weight = 0.02
        
        total_loss = content_loss_weight*content_loss + style_loss_weight*style_loss
        
        return content_loss, style_loss, total_loss

def load_image(image_dir, height, width):
    image = cv2.imread(image_dir)
    resized_image = cv2.resize(image, (width,height))
    return resized_image[np.newaxis,...]

def apply_pixel_correction(img, bgr_values, sign):
    fix = np.array(bgr_values).reshape((1,1,1,3))
    if sign == '+':
        return img + fix
    else:
        return img - fix

def apply_noise(image, noise_ratio=0.6):
    height = image.shape[1]
    width = image.shape[2]
    noise_image = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + image * (1 - noise_ratio)


def display_bgr_image(img):
    #img is in BGR format
    #cv2.cvtColor function gives error - because of numpy matrix dtype?
    rgb_image = copy.deepcopy(img)
    rgb_image = rgb_image[0,...]
    #https://stackoverflow.com/questions/4661557/pil-rotate-image-colors-bgr-rgb
    rgb_image = rgb_image[:,:,::-1]
    plt.imshow(rgb_image)

def save_image(img, save_dir):
    cv2.imwrite(output_dir+str(i)+"GEN_IMAGE.png", gen_image[0,...])

sess = tf.InteractiveSession()

vgg_link = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat' #~500mb!
#C:\\Users\\Pavitrakumar\\Desktop\\
vgg_model_name = 'imagenet-vgg-verydeep-19.mat'
#download(vgg_link, vgg_model_name)
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
content_layer = 'conv4_2'

style_image_dir = "C:\\Users\\Pavitrakumar\\Desktop\\starry_night.jpg"
content_image_dir = "C:\\Users\\Pavitrakumar\\Desktop\\sample2.jpg"
output_dir = "C:\\Users\\Pavitrakumar\\Desktop\\NST\\"
img_height = 400
img_width = 600
image_channels = 3
custom_pix = [123.68, 116.779, 103.939]

with tf.variable_scope('input'):
    image_placeholder = tf.Variable(np.zeros([1, img_height, img_width, image_channels]), dtype=tf.float32, name = "image_placeholder")

#image_placeholder -> is not actually a placeholder tf variable, but a common variable for both input image(the image
#we want to stylize) and the style image (a painting or a pattern) - we are [[training]] the input image to look more
#like the style image - so it needs to be init. as a VARIABLE not a placeholder tf class.

#below gives error (negative dim size) for any dim(w or h) <= 250 - for padding = valid
#padding = same (for conv2d and pooling) gives no error!

model = load_vgg(vgg_model_name, image_placeholder)

content_image = load_image(content_image_dir,img_height,img_width)
#display_bgr_image(content_image1)
content_image = apply_pixel_correction(content_image,custom_pix,'-')
#display_bgr_image(content_image)

style_image = load_image(style_image_dir,img_height,img_width)
#display_bgr_image(style_image)
style_image = apply_pixel_correction(style_image,custom_pix,'-')
#display_bgr_image(style_image)

img_after_noise = apply_noise(content_image)
#display_bgr_image(img_after_noise)


#training stuff

_content_loss, _style_loss, _total_loss = get_combined_loss(sess, image_placeholder, content_image, style_image, model, style_layers, content_layer)

optimizer = tf.train.AdamOptimizer(2).minimize(_total_loss)


sess.run(tf.global_variables_initializer())

sess.run(image_placeholder.assign(img_after_noise)) #TO BE DONE AFTER INIT!!! PUTTING IT BEFORE GLOBAL INIT DOES NOT GIVE PROPER IMAGE!

#writer = tf.summary.FileWriter('logs', sess.graph)

skip_step = 1
start_time = time.time()
for i in range(300):
    #get generated image
    sess.run(optimizer)
    if i % 5 == 0:
        gen_image, loss = sess.run(fetches = [image_placeholder, _total_loss])
        print("step: ",i,"loss: ",loss)
        #print(gen_image.shape)
        gen_image = apply_pixel_correction(gen_image,custom_pix,'+')
        #display_bgr_image(gen_image)
        cv2.imwrite(output_dir+str(i)+"GEN_IMAGE.png", gen_image[0,...])
        diff_time = time.time() - start_time
        print("time taken for last step:",diff_time)
        start_time = time.time()
