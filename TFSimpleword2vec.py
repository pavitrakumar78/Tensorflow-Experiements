# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:33:21 2017

@author: Pavitrakumar
"""

#http://web.stanford.edu/class/cs20si/lectures/notes_04.pdf
#http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb
#https://github.com/chiphuyen/tf-stanford-tutorials/blob/master/examples/04_word2vec_no_frills.py
#https://github.com/kingtaurus/cs224d/blob/master/assignment1/q3_word2vec_sol.py
#https://stackoverflow.com/questions/41475180/understanding-tf-nn-nce-loss-in-tensorflow
#https://ronxin.github.io/wevi/
#https://stackoverflow.com/questions/41860871/how-to-use-tflearn-deep-learning-for-document-classification/41862567

"""
CODE NOTE TESTED COMPLETELY!
"""

import tensorflow as tf
import zipfile
from collections import Counter
import numpy as np
from sklearn.preprocessing import normalize

tf.reset_default_graph()
sess = tf.InteractiveSession()

def read_data(file_path):
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split() 
        # tf.compat.as_str() converts the input into the string
    return words


all_words = read_data("C:\\Users\\Pavitrakumar\\Desktop\\text8.zip")
#above is text returned as a list of words
#output for above line: ['above', 'is', 'text', 'returned', 'as', 'a', 'list', 'of', 'words' ...]

#build vocabulary and a dictionary!
vocab_size = 50000
dictionary = dict()
count = [('UNK', -1)]
#we extract top 10,000 most common words
count.extend(Counter(all_words).most_common(vocab_size - 1))
#now we build a dictionary
for index,word_count_pair in enumerate(count):
    dictionary[word_count_pair[0]] = index

"""
[('UNK', -1),
 ('the', 1061396),
 ('of', 593677),
 ('and', 416629),
 ('one', 411764),
 ('in', 372201),
 ('a', 325873),
 ('to', 316376),
 ('zero', 264975),
 ('nine', 250430)]
"""

#now to build the input data, we need to one-hot encode the words using the dictionary we have
#ex> if words are : ['one', 'of', 'the', 'seven', 'wonders', 'of', ... ] using the above dictionary ONLY (sample), 
#     it would be : [4, 2, 1, 0, 0, 2, ...]

word_indices = [dictionary[word] if word in dictionary else 0 for word in all_words]

#now, we are building a skip-gram model, so we need word-pairs as described here 
#http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

def gen_skip_model_pairs(word_indices, max_context_window_size, num_pairs, start_index = 0):
    #start_index -> where we want to start making the the pairs
    pairs = []
    center_words = []
    target_words = []
    current_index = start_index
    max_len = len(word_indices)
    while(current_index<max_len and len(pairs)<num_pairs):
        #approximately select num_pairs/2 pairs using context words before center_word and
        #num_pairs/2 pairs using context words after center_word
        center_word = word_indices[current_index]
        n = np.random.randint(1,max_context_window_size)
        #randomly select n words from left half
        #n = randrange(leftmost index,center_word index)
        #i.e if window is 5, randomly select 1-5 words from left side
        for target in word_indices[max(0, current_index - n) : current_index]:
            pairs.append((center_word, target))
            center_words.append(center_word)
            target_words.append(target)
        
        for target in word_indices[current_index + 1 : current_index + n + 1]:
            pairs.append((center_word, target))
            center_words.append(center_word)
            target_words.append(target)
        
        current_index = current_index + 1
    #in-case we have more than we need
    return center_words[0:num_pairs], target_words[0:num_pairs], pairs[0:num_pairs],current_index
            
        

def construct_network(inp_shape, out_shape, hidden_dims = [128], activation = tf.nn.relu):
    #Simple Multi-layer-NN - Only dense layers - no conv
    with tf.variable_scope("nnet"): #not really necessary - for clarity here. Maybe useful in specific cases where we reuse vars
        X = tf.placeholder(shape = [None, inp_shape], dtype = tf.float32) #don't go lower than float32; else will get 'nan' in weights
        Y = tf.placeholder(shape = [None, out_shape], dtype = tf.float32)
        
        net = X
        for index,layer_size in enumerate(hidden_dims):
            with tf.variable_scope('layer{}'.format(index)): #not really necessary - for clarity here. Maybe useful in specific cases where we reuse vars
                #acces using nnet/layer{0,1,2..}
                net = tf.layers.dense(inputs = net, units = layer_size, activation = activation, name = 'net{}'.format(index))
                #net = tf.layers.batch_normalization(net) #improves acc? - possibly.
        
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


def word2vec_model(vocab_size, embed_size, batch_size):
    center_words_var = tf.placeholder(tf.int32, shape=[batch_size], name='center_words_var')
    target_words_var = tf.placeholder(tf.int32, shape=[batch_size, 1], name='target_words_var')

    # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
    # Step 2: define weights. In word2vec, it's actually the weights that we care about
    embed_matrix = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0), 
                            name='embed_matrix')

    # Step 3: define the inference
    embed = tf.nn.embedding_lookup(embed_matrix, center_words_var, name='embed')

    # Step 4: construct variables for NCE loss
    nce_weight = tf.Variable(tf.truncated_normal([vocab_size, embed_size],
                                                stddev=1.0 / (embed_size ** 0.5)), 
                                                name='nce_weight')
    nce_bias = tf.Variable(tf.zeros([vocab_size]), name='nce_bias')

    # define loss function to be NCE loss function
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                        biases=nce_bias, 
                                        labels=target_words_var, 
                                        inputs=embed, 
                                        num_sampled=64, 
                                        num_classes=vocab_size), name='loss')

    # Step 5: define optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    return loss, optimizer, embed_matrix, center_words_var, target_words_var


center_words, target_words,pairs,ind = gen_skip_model_pairs(word_indices,max_context_window_size = 5,num_pairs = 10,start_index = 0)


#now the input to NN is a 1hot encoded version of the center_words and the output is the 1hot encoded version of the target words
#suppose center is 12, and vocabulary size is 20, 1hot encoded vec will be [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
#suppose target is 2, and vocabulary size is 20, 1hot encoded vec will be [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

del all_words

def one_hot_encode_words(word_indices,vocab_size):
    X = np.zeros((len(word_indices),vocab_size))
    X[list(range(len(word_indices))),word_indices] = 1
    return X

X_train = one_hot_encode_words(center_words,vocab_size)
Y_train = one_hot_encode_words(target_words,vocab_size)

#NN-related
X, Y, train_expr, loss, pred_class, accuracy, logits = construct_network(inp_shape = vocab_size, 
                                                                         out_shape = vocab_size,
                                                                         hidden_dims = [128])

#loss, optim, embed_mat, X, Y = word2vec_model(vocab_size, embed_size = 128, batch_size = 128)

sess.run(tf.global_variables_initializer())
ind = 0
batch_size = 128

for i in range(10000):
    center_words, target_words,pairs,ind = gen_skip_model_pairs(word_indices,max_context_window_size = 5,num_pairs = batch_size,start_index = ind)
    X_train = one_hot_encode_words(center_words,vocab_size)
    Y_train = one_hot_encode_words(target_words,vocab_size)
    
    #Using NN  - not sure if this works..
    acc, _loss, _logits, _ = sess.run(fetches = [accuracy, loss, logits, train_expr],feed_dict = {X: X_train, Y: Y_train})
    embeddings = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nnet/layer0/')[0].eval() #(50000, 128)
    
    #Using proper word2vec algos
    #_loss, _logits, emb = sess.run(fetches = [loss, optim, embed_mat],feed_dict = {X: center_words, Y: np.array(target_words).reshape((batch_size,1))})    
    #embeddings = emb
    
    #embeddings = embeddings / np.linalg.norm(embeddings)


    #print(np.sum(embeddings))
    print(" Training Loss: ",_loss)
    #testing similary between 'of' and 'the'
    #we already have embeddings
    """
    word1_vector = one_hot_encode_words([dictionary['of']],vocab_size) # (1, 50000)
    word2_vector = one_hot_encode_words([dictionary['the']],vocab_size) # (1, 50000)
        
        
    word1_embedding = np.dot(word1_vector,embeddings)
    word2_embedding = np.dot(word2_vector,embeddings)
        
    similarity = np.dot(word1_embedding,word2_embedding.T)
    print("similarity between 'of' and 'the'",similarity)
    """

#embeddings = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nnet/layer0/')[0]
