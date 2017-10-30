"""
    Act_classification-with-CNN/model.py
    
    | Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks [Lee,2016]   
    
    
    | CNN based short-text representation and Sequential short-text classification model of this paper is implemented using tensorflow.
        
    | This model comprises two parts. 
    * The first part generates a vector representation for each short text using CNN architecture.
    * The second part classifies the current short text based on the vector representations of current as well as a few preceding short texts.  
    | The components of the model corresponding to each code are specified as comment.
    
    .. figure:: model.jpg
       :align: center
       
       This is whole model diagram. 
    

"""
import math
import numpy as np
import tensorflow as tf



class Model(object):
    """
    A Convolutional Neural Network(CNN) for text representation,
    A Feedforward Neural Network(FNN) for text classification
    
    | CNN : Use an embedding layer, followed by a convolutional, max-pooling and ReLU layer.
    | FNN : Use tanh layer, followed by a sigmoid layer
    
    
    :param int max_sentence_len: After the sentence preprocessing of the training data, the length of the longest sentence 
    :param int num_classes: The number of classes in the output layer (the number of Acts)
    :param int vocab_size: Vocabulary size of word vector
    :param int embed_size: Dimension of word vector
    :param int filter_size: Filter size CNN layer
    :param int num_filters: Number of filters of CNN layers
    :param int history_size1: History size of first FNN layer
    :param int history_size2: History size of second FNN layer
    """
    
    def __init__(
        self, max_sentence_len, num_classes, vocab_size, embed_size,
        filter_size, num_filters, history_size1, history_size2):
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [history_size1 + history_size2 + 1, max_sentence_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [1, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 
    
        # Embedding layer (pre-trained word vector)   
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            """Embedding layer (pre-trained word vector)
            """
            self.W = tf.Variable( tf.random_uniform([vocab_size, embed_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
         
            # Make it a 4D tensor
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            
        # Convolution layer
        with tf.name_scope("CNN"):
            # Create a convolution + maxpool layer for filter size
            filter_shape = [filter_size, embed_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="weight")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="biases")
            
            conv = tf.nn.conv2d(self.embedded_chars_expanded, 
                                filter=W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
            
            # Apply nonlinearity
            c = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(value=c,
                                    ksize=[1, max_sentence_len - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="pool")
           
        # Add dropout
        with tf.name_scope("dropout"):
            self.c_drop = tf.nn.dropout(pooled, self.dropout_keep_prob)
        
        sentence_list = tf.squeeze(self.c_drop)
 

        # FNN layer
        with tf.name_scope('FNN1'):
            FNN1_list =[]    
            def fnn1(input,weight_shape,bias_shape):
                #create variable named "w"
                w = []
                for weight_index in range(history_size1+1):
                    weight_name = "w_"+str(weight_index)
                    w.append(tf.get_variable(weight_name, weight_shape, initializer = tf.random_normal_initializer()))
                
                # Create variable named  "b1"
                b1 = tf.get_variable("b1", bias_shape,
                                     initializer = tf.constant_initializer(0.0))
                
                # sum of wx and b1
                output = np.zeros(weight_shape[1],dtype=np.float32)          
                for i in range(history_size1,-1,-1):
                    temp = tf.matmul(tf.reshape(input[i-history_size1],shape=[1,weight_shape[0]]),w[i])
                    output = tf.add(output,temp)
                    
                return tf.nn.tanh(tf.nn.bias_add(output,b1))
            
            
            for i in range(history_size2+1):
                    if i is 0:
                        with tf.variable_scope("FNN1"):
                            FNN1_list.append(fnn1(input = sentence_list[i:i + history_size1+1],
                                                  weight_shape = [num_filters, num_classes],
                                                  bias_shape = [num_classes]))
                    elif i is not 0:
                        with tf.variable_scope("FNN1",reuse=True):
                            FNN1_list.append(fnn1(input = sentence_list[i:i + history_size1+1],
                                                  weight_shape = [num_filters, num_classes],
                                                  bias_shape = [num_classes]))
                       
            
        # sigmoid
        with tf.name_scope('FNN2'):
            #create variable named "w"           
            w = []
            for weight_index in range(history_size2 + 1):
                weight_name = "w_"+str(weight_index)
                w.append(tf.Variable(tf.random_normal([num_classes,num_classes], stddev=0.35),name=weight_name))
            
            # Create variable named  "b2"
            b2 = tf.Variable(tf.zeros([num_classes]), name="b2")
            
            # sum of wx and b1
            output = np.zeros(num_classes,dtype=np.float32)
            for i in range (history_size2,-1,-1):
                         temp = tf.matmul(FNN1_list[i - history_size2], w[i])
                         output = tf.add(output,temp)
            
            FNN2_output = tf.nn.bias_add(output,b2)
            
        with tf.name_scope("output"):
            self.scores = tf.nn.sigmoid(FNN2_output, name='sigmoid')
            self.prediction = tf.round(self.scores, name='predictions')
    
        with tf.name_scope("loss"):  
            # This is like sigmoid_cross_entropy_with_logits() except that pos_weight for the positive targets
            losses = tf.nn.weighted_cross_entropy_with_logits(logits = FNN2_output,
                                                              targets = self.input_y,
                                                              pos_weight = 4,
                                                              name = 'cross_entropy')
            self.loss=tf.reduce_mean(losses, name='cross_entropy_mean')
                      
        with tf.name_scope("accuracy"):
            self.label = tf.round(self.input_y)
            correct_prediction = tf.equal(self.prediction, self.label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                           name="accuracy")
 
