import tensorflow as tf
import numpy as np

class Cnn(object):    
    def weight_variable(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W, strides=[1, 1, 1, 1], padding='VALID', name=None):
        return tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)

    def max_pool_2x2(self, x, ksize, strides=[1,2,2,1], padding='VALID', name=None):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, name=name)

    def various_accuracy(self, num_labels=None, y_input_index=None, y_pre_index=None):
        if num_labels == None or y_input_index == None or y_pre_index == None:
            return None
        
        #输入次数 预测次数(含总、对和错) 正确次数 正确率
        accuracys = []
        for i in range(num_labels+1):
            accuracys.append([0,0,0,0])
        for i in range(len(y_input_index)):
            accuracys[y_input_index[i]][0] += 1
            accuracys[y_pre_index[i]][1] += 1

            accuracys[-1][0] += 1
            accuracys[-1][1] += 1

            if y_input_index[i] == y_pre_index[i]:
                accuracys[y_input_index[i]][2] += 1
                accuracys[-1][2] += 1

        for i in range(len(accuracys)):
            if accuracys[i][0] != 0:
                accuracys[i][3] = float(accuracys[i][2]) / accuracys[i][0]

        return accuracys


class CnnText(Cnn):
    def __init__(
        self, sequence_length, num_classes,
        embedding_size, filter_sizes, num_filters, full_layer_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output, dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name = "input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer        
        # self.embedded_chars = [None(batch_size), sequence_size, embedding_size, 1(num_channels)]
        self.embedded_chars = self.input_x
        self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters_total = 0
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = self.weight_variable(filter_shape, "W")
                b = self.bias_variable([num_filters], "b")                
                conv = self.conv2d(self.embedded_chars_expended, W, name="conv")
 
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
            pooled = self.max_pool_2x2(h, [1, 2, 1, 1], strides=[1,2,2,1], name='pool')
            

            hh = (sequence_length - filter_size + 1) // 2
            pooled = tf.reshape(pooled, [-1, num_filters, hh * 1])
            num_filters_total += hh * 1 * num_filters
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.h_pool = tf.concat(pooled_outputs, 2)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # full layer
        W_fc1 = self.weight_variable([num_filters_total, full_layer_filters])
        b_fc1 = self.bias_variable([full_layer_filters])
        self.full_layer = tf.nn.relu(tf.matmul(self.h_pool_flat, W_fc1) + b_fc1)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.full_layer, self.dropout_keep_prob)
        
        # Final (unnomalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
            "W",
            shape = [full_layer_filters, num_classes],
            initializer = tf.contrib.layers.xavier_initializer())
        b = self.bias_variable([num_classes], "b")        
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)

        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")
        self.predictions = tf.argmax(self.scores, 1, name = "predictions")
        self.input_y_index = tf.argmax(self.input_y, 1, name = "input_y_index")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y_index)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")

class CnnImage(Cnn):
    def __init__(
        self, sequence_length, num_classes,
        embedding_size, filter_sizes, num_filters, full_layer_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output, dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name = "input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer        
        # self.embedded_chars = [None(batch_size), sequence_size, embedding_size, 1(num_channels)]
        self.embedded_chars = self.input_x
        self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters_total = 0
        for size in filter_sizes:
            filter_shape = [size, size, 1, num_filters]
            W = self.weight_variable(filter_shape, "W")
            b = self.bias_variable([num_filters], "b")                
            conv = self.conv2d(self.embedded_chars_expended, W, name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
            pooled = self.max_pool_2x2(h, [1, 2, 2, 1], name='pool')

            hh = (sequence_length - size + 1) // 2
            ll = (embedding_size - size + 1) // 2
            pooled = tf.reshape(pooled, [-1, num_filters, hh * ll])
            num_filters_total += hh * ll * num_filters
            pooled_outputs.append(pooled)

        h_pool = tf.concat(pooled_outputs, 2)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # full layer
        W_fc1 = self.weight_variable([num_filters_total, full_layer_filters])
        b_fc1 = self.bias_variable([full_layer_filters])
        self.full_layer = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.full_layer, self.dropout_keep_prob)
        
        # Final (unnomalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
            "W",
            shape = [full_layer_filters, num_classes],
            initializer = tf.contrib.layers.xavier_initializer())
        b = self.bias_variable([num_classes], "b")

        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)

        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")
        self.predictions = tf.argmax(self.scores, 1, name = "predictions")
        self.input_y_index = tf.argmax(self.input_y, 1, name = "input_y_index")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y_index)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")