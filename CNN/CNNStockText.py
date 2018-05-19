#coding=utf-8
import os
import tensorflow as tf
import numpy as np
import time
import datetime
import pandas as pd
from .DataHelper import DataHelper
from .Cnn import CnnText
from .Cnn import CnnImage

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Spider.tool import VTool

class CNNStockText():
    def __init__(self,):
        self.num_labels = 2

        # Model hyperparameters
        self.embedding_dim = 64
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 8
        self.dropout_keep_prob = 0.8
        self.l2_reg_lambda = 0.1
        self.full_layer_filters = 64
        self.learn_rate = 0.008

        # Training paramters
        self.batch_size = 10
        # self.num_epochs = 200

        # Misc parameters
        self.allow_soft_placement = True # 遇到无法用GPU跑的数据时，自动切换成CPU进行
        self.log_device_placement = False # 记录日志

        self.max_word_num = 6000
        self.blank_word = ''
    
    def train(self, basic_path=None, input_file=None, output_folder=None, word2vec_model=None, filter_sizes=None, folder_extra='', reduce_num=0, test_part_start=0.9, test_part_end=1, times=10):
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if input_file is None or output_folder is None:
            return None

        word2vec_load = None
        if word2vec_model is not None:
            word2vec_load = os.path.join(basic_path, word2vec_model)
        
        input_path = os.path.join(basic_path, input_file)
        output_path = os.path.join(basic_path, output_folder)
        VTool.makeDirs(folders=[output_path])

        if filter_sizes != None:
            self.filter_sizes = filter_sizes

        print("Writing to {}\n".format(output_path))
        tf.reset_default_graph()
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement = self.allow_soft_placement,
        	log_device_placement = self.log_device_placement)
            sess = tf.Session(config = session_conf)
            with sess.as_default():
                x_train, y_train, batch_index, max_word_num = DataHelper.get_text_train_data(file=input_path, batch_size=self.batch_size, reduce_num=reduce_num, test_part_start=test_part_start, test_part_end=test_part_end)
                self.max_word_num = max_word_num if max_word_num < self.max_word_num else self.max_word_num
                if len(y_train) <= 0:
                    print("CSV No Data!!!")
                    exit()
                sentences, max_document_length = DataHelper.padding_sentences(x_train, self.blank_word, self.max_word_num)
                x_train = np.array(DataHelper.embedding_sentences(sentences, embedding_size = self.embedding_dim, file_to_load = word2vec_load, file_to_save = word2vec_load))
                del sentences                
                print("x.shape = {}".format(x_train.shape))
                print("y.shape = {}".format(y_train.shape))                

                cnn = CnnText(
                    sequence_length = x_train.shape[1],
                    num_classes = y_train.shape[1],
                    embedding_size = self.embedding_dim,
                    filter_sizes = self.filter_sizes,
                    num_filters = self.num_filters,
                    full_layer_filters = self.full_layer_filters,
                    l2_reg_lambda = self.l2_reg_lambda)

        	    # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(self.learn_rate)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                '''
                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                print("Writing to {}\n".format(output_path))
                
                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                summary_dir = os.path.join(output_path, "train_summaries")
                summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
                VTool.makeDirs(folders=[summary_dir])
                '''

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(output_path, "checkpoints"+folder_extra))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")

                VTool.makeDirs(folders=[checkpoint_dir])
                saver = tf.train.Saver(tf.global_variables())

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                for i in range(times):
                    for step in range(len(batch_index)-1):
                        feed_dict = {
                            cnn.input_x: x_train[batch_index[step]:batch_index[step+1]],
                            cnn.input_y: y_train[batch_index[step]:batch_index[step+1]],
                            cnn.dropout_keep_prob: self.dropout_keep_prob
                        }

                        _, step, loss, accuracy, predictions, input_y_index = sess.run(
                            [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions, cnn.input_y_index],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                        all_accuracy = cnn.various_accuracy(self.num_labels, input_y_index.tolist(), predictions.tolist())
                        for a in all_accuracy:
                            print("input_nums: {:g}, pre_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(a[0], a[1], a[2], a[3]))
                        # summary_writer.add_summary(summaries, step)

                    if i % 5 == 0:
                        print("保存模型：", saver.save(sess, checkpoint_prefix))
                print("保存模型：", saver.save(sess, checkpoint_prefix))
                print("The train has finished")

    def predict(self, basic_path=None, input_file=None, word2vec_model=None, output_folder=None, filter_sizes=None, folder_extra='', reduce_num=0, test_part_start=0.9, test_part_end=1):
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if input_file is None or output_folder is None:
            return None

        input_path = os.path.join(basic_path, input_file)
        output_path = os.path.join(basic_path, output_folder)

        word2vec_load = None
        if word2vec_model is not None:
            word2vec_load = os.path.join(basic_path, word2vec_model)

        if filter_sizes != None:
            self.filter_sizes = filter_sizes
        
        x_test, y_test, y_others, max_word_num = DataHelper.get_text_test_data(file=input_path, reduce_num=reduce_num, test_part_start=test_part_start, test_part_end=test_part_end)
        self.max_word_num = max_word_num if max_word_num < self.max_word_num else self.max_word_num
        if len(y_test) <= 0:
            print("CSV No Data!!!")
            exit()
        sentences, max_document_length = DataHelper.padding_sentences(x_test, self.blank_word, self.max_word_num)
        x_test = np.array(DataHelper.embedding_sentences(sentences, embedding_size = self.embedding_dim, file_to_load=word2vec_load))
        del sentences

        print("x.shape = {}".format(x_test.shape))
        print("y.shape = {}".format(y_test.shape))

        tf.reset_default_graph()
        cnn = CnnText(
            sequence_length = x_test.shape[1],
            num_classes = y_test.shape[1],
            embedding_size = self.embedding_dim,
            filter_sizes = self.filter_sizes,
            num_filters = self.num_filters,
            full_layer_filters = self.full_layer_filters,
            l2_reg_lambda = self.l2_reg_lambda)
        saver=tf.train.Saver()

        with tf.Session() as sess:
            #参数恢复
            checkpoint_dir = os.path.abspath(os.path.join(output_path, "checkpoints"+folder_extra))
            module_file = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, module_file)
            
            all_predictions = []
            all_input = []

            batch_size = 50
            cur_len, all_len = 0, len(x_test)
            while cur_len < all_len:
                cur_end = cur_len + batch_size if cur_len + batch_size <= all_len else all_len

                feed_dict = {
                    cnn.input_x: x_test[cur_len: cur_end],
                    cnn.input_y: y_test[cur_len: cur_end],
                    cnn.dropout_keep_prob: 1.0
                }

                loss, accuracy, predictions, input_y_index = sess.run(
                    [cnn.loss, cnn.accuracy, cnn.predictions, cnn.input_y_index],
                    feed_dict)
                
                print("len {} to {}, loss {:g}, acc {:g}".format(cur_len, cur_end, loss, accuracy))
                cur_len = cur_end
                all_predictions.extend(predictions.tolist())
                all_input.extend(input_y_index.tolist())

            all_accuracy = cnn.various_accuracy(self.num_labels, all_input, all_predictions)
            for a in all_accuracy:
                print("input_nums: {:g}, pre_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(a[0], a[1], a[2], a[3]))

        profit, origin_profit = DataHelper.rateCalc(all_predictions, y_others, self.num_labels-1)
        return all_accuracy[-1][3], profit, origin_profit, all_predictions, y_others