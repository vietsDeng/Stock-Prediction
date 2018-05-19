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

class CNNStockNumber():
    def __init__(self,):
        self.num_labels = 2

        self.embedding_dim = None
        self.filter_sizes = [2]
        self.filter_embeddings = None
        self.num_filters = 16
        self.dropout_keep_prob = 0.8
        self.l2_reg_lambda = 0.1
        self.full_layer_filters = 1024
        self.learn_rate = 0.001
                
        self.batch_size = 10

        # self.num_epochs = 200
        # self.num_checkpoints = 1
        
        self.allow_soft_placement = True
        self.log_device_placement = False
    
    def train(self, basic_path=None, input_file=None, output_folder=None, embedding_dim=0, folder_extra='', filter_sizes=None, reduce_num=0, test_part_start=0.9, test_part_end=1, data_stand=False, times=10):
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if input_file is None or output_folder is None:
            return None
        if filter_sizes is not None:
            self.filter_sizes = filter_sizes
        
        input_path = os.path.join(basic_path, input_file)
        output_path = os.path.join(basic_path, output_folder)
        VTool.makeDirs(folders=[output_path])
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        print("Writing to {}\n".format(output_path))
        
        tf.reset_default_graph()
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement = self.allow_soft_placement,
            log_device_placement = self.log_device_placement)
            sess = tf.Session(config = session_conf)
            with sess.as_default():
                x_train, y_train, batch_index, _, _, _ = DataHelper.get_number_data(file=input_path, batch_size=self.batch_size, reduce_num=reduce_num, test_part_start=test_part_start, test_part_end=test_part_end, stand=data_stand)
                if len(x_train) <= 0:
                    print("CSV No Data!!!")
                    exit()                
                print("x.shape = {}".format(x_train.shape))
                print("y.shape = {}".format(y_train.shape))

                cnn = CnnImage(
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

                checkpoint_dir = os.path.abspath(os.path.join(output_path, "checkpoints"+folder_extra))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                
                VTool.makeDirs(folders=[checkpoint_dir])
                saver = tf.train.Saver()                
                sess.run(tf.global_variables_initializer())
                for i in range(times):
                    for step in range(len(batch_index)-1):
                        feed_dict = {
                            cnn.input_x: x_train[batch_index[step]:batch_index[step+1]],
                            cnn.input_y: y_train[batch_index[step]:batch_index[step+1]],
                            cnn.dropout_keep_prob: self.dropout_keep_prob
                        }

                        _, loss, accuracy, predictions, input_y_index = sess.run(
                            [train_op, cnn.loss, cnn.accuracy, cnn.predictions, cnn.input_y_index],
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

    def predict(self, basic_path=None, input_file=None, output_folder=None, embedding_dim=1, folder_extra='', filter_sizes=None, reduce_num=0, test_part_start=0.9, data_stand=False, test_part_end=1):
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if input_file is None or output_folder is None:
            return None
        if filter_sizes is not None:
            self.filter_sizes = filter_sizes

        input_path = os.path.join(basic_path, input_file)
        output_path = os.path.join(basic_path, output_folder)     
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        
        _, _, _, x_test, y_test, y_others = DataHelper.get_number_data(file=input_path, batch_size=self.batch_size, reduce_num=reduce_num, test_part_start=test_part_start, test_part_end=test_part_end, stand=data_stand)
        if len(y_test) <= 0:
            print("CSV No Data!!!")
            exit()        
        print("x.shape = {}".format(x_test.shape))
        print("y.shape = {}".format(y_test.shape))

        tf.reset_default_graph()
        cnn = CnnImage(
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
        
            feed_dict = {
                cnn.input_x: x_test,
                cnn.input_y: y_test,
                cnn.dropout_keep_prob: 1.0
            }
            
            loss, accuracy, predictions, input_y_index = sess.run(
                [cnn.loss, cnn.accuracy, cnn.predictions, cnn.input_y_index],
                feed_dict)

            print("All: loss {:g}, acc {:g}".format(loss, accuracy))
            all_accuracy = cnn.various_accuracy(self.num_labels, input_y_index.tolist(), predictions.tolist())
            for a in all_accuracy:
                print("input_nums: {:g}, pre_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(a[0], a[1], a[2], a[3]))

        profit, origin_profit = DataHelper.rateCalc(predictions, y_others, self.num_labels-1)
        return accuracy, profit, origin_profit, predictions, y_others