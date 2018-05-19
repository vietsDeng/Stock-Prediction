#coding=utf-8

from CNN.CNNStockText import CNNStockText
from CNN.CNNStockNumber import CNNStockNumber
from CNN.DataHelper import DataHelper

from LSTM.LSTMStockOrigin import LSTMStockOrigin

from Spider.tool import VTool

import os
import numpy as np
import pandas as pd
import time
import random

import tensorflow as tf
import datetime

import gc

# conn, cur = VMysql.get_mysql()

class Ensemble(object):    
    def __init__(self,):
        self.basic_path = 'D:\\workspace\\Mine\\run_data'
        self. learn_rate = 0.01
        self.drop = 0.05

    def dateCompare(self, item1, item2):
        t1 = time.mktime(time.strptime(item1, '%Y-%m-%d'))
        t2 = time.mktime(time.strptime(item2, '%Y-%m-%d'))        
        if t1 < t2:
            return -1
        elif t1 > t2:
            return 1
        else:
            return 0
        
    def makeEnsembleData(self, stock_folders=None):
        test_part_array = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        test_part_times = 10
        is_reload = False

        folder_600104 = "600104/" #3
        folder_601318 = "601318/" #5
        folder_002230 = "002230/" #7

        basic_path = self.basic_path

        stock_folders = stock_folders if stock_folders != None else [folder_600104]
        for choose_stock_folder in stock_folders:
            for i in range(len(test_part_array)-1):
                for j in range(test_part_times):
                    
                    folder_extra = '_' + str(i) + '_' + str(j)
                    data_file = os.path.join(basic_path, choose_stock_folder, "ensemble/checkpoints" + folder_extra, "ensemble_data.csv")
                    if os.path.exists(data_file) and not is_reload:
                        continue
                    else:
                        VTool.makeDirs(files=[data_file])

                    data = []

                    # cnn-model-1 1
                    cst1 = CNNStockText()
                    accuracy, profit, origin_profit, predictions, others = cst1.predict(basic_path=basic_path, input_file=choose_stock_folder+"cnn/title_data.csv", output_folder=choose_stock_folder+"cnn/title_run", word2vec_model="news_title_word2vec", filter_sizes=[3, 4, 5], folder_extra=folder_extra, reduce_num=21, test_part_start=test_part_array[0], test_part_end=test_part_array[-1])
                    data.append([accuracy, profit, origin_profit, predictions, others])

                    cst2 = CNNStockText()
                    # cnn-model-2 2
                    accuracy, profit, origin_profit, predictions, others = cst2.predict(basic_path=basic_path, input_file=choose_stock_folder+"cnn/tfidf_text_data.csv", output_folder=choose_stock_folder+"cnn/text_run", word2vec_model="news_tfidf_word2vec", filter_sizes=[8, 9, 10], folder_extra=folder_extra, reduce_num=21, test_part_start=test_part_array[0], test_part_end=test_part_array[-1])
                    data.append([accuracy, profit, origin_profit, predictions, others])

                    # cnn-model-3 3
                    csn = CNNStockNumber()            
                    accuracy, profit, origin_profit, predictions, others = csn.predict(basic_path=basic_path, input_file=choose_stock_folder+"cnn/bindex_data.csv", output_folder=choose_stock_folder+"cnn/bindex_run", embedding_dim=3, filter_sizes=[2], folder_extra=folder_extra, reduce_num=21, test_part_start=test_part_array[0], test_part_end=test_part_array[-1])
                    data.append([accuracy, profit, origin_profit, predictions, others])

                    # cnn-model-4 4
                    accuracy, profit, origin_profit, predictions, others = csn.predict(basic_path=basic_path, input_file=choose_stock_folder+"cnn/news_stock_data_%s.csv" % i, output_folder=choose_stock_folder+"cnn/news_stock_run", embedding_dim=10, filter_sizes=[3, 4, 5], folder_extra=folder_extra, reduce_num=0, test_part_start=test_part_array[0], test_part_end=test_part_array[-1])
                    data.append([accuracy, profit, origin_profit, predictions, others])
                    
                    # lstm-model-1 5
                    so = LSTMStockOrigin()
                    accuracy, profit, origin_profit, predictions, others = so.predict_rate(basic_path=basic_path, data_file=choose_stock_folder+"lstm/stock_origin_data.csv", model_folder=choose_stock_folder+"lstm/origin_model", folder_extra=folder_extra, reduce_num=10, test_part_start=test_part_array[0], test_part_end=test_part_array[-1])
                    data.append([accuracy, profit, origin_profit, predictions, others])

                    # lstm-model-2 6
                    accuracy, profit, origin_profit, predictions, others = so.predict_rate(basic_path=basic_path, data_file=choose_stock_folder+"lstm/stock_bindex_data.csv", model_folder=choose_stock_folder+"lstm/bindex_model", folder_extra=folder_extra, reduce_num=10, test_part_start=test_part_array[0], test_part_end=test_part_array[-1])
                    data.append([accuracy, profit, origin_profit, predictions, others])

                    # lstm-model-3 7
                    accuracy, profit, origin_profit, predictions, others = so.predict_rate(basic_path=basic_path, data_file=choose_stock_folder+"lstm/stock_news_data_%s.csv" % i, model_folder=choose_stock_folder+"lstm/news_model", folder_extra=folder_extra, reduce_num=10, test_part_start=test_part_array[0], test_part_end=test_part_array[-1])
                    data.append([accuracy, profit, origin_profit, predictions, others])

                    ensemble_data = {}
                    for d in data:
                        for di in range(len(d[3])):
                            if d[4][di][1] not in ensemble_data:
                                ensemble_data[d[4][di][1]] = [d[4][di][1], d[4][di][0], []]
                            
                            ensemble_data[d[4][di][1]][2].append(d[3][di])
                    
                    data_len = len(data)
                    data = {}
                    for k in ensemble_data:
                        d = ensemble_data[k]
                        if len(d[2]) == data_len:
                            data[k] = d

                    e_data = sorted(data.items(), key=lambda x: time.mktime(time.strptime(x[0], '%Y-%m-%d')))
                    data = {"date": [], "rate": [], "predictions": []}
                    for d in e_data:
                        data["date"].append(d[1][0])
                        data["rate"].append(d[1][1])
                        data["predictions"].append(d[1][2])
                    pd.DataFrame(data).to_csv(data_file, index=False, columns=["date", "rate", "predictions"])
                    del cst1, cst2, e_data, data
                    gc.collect()
                    #exit()

    @classmethod
    def classify(cls, y):
        y = float(y)
        if y <= 0:
            return [1,0]
        else:
            return [0,1]

    @classmethod
    def get_data(cls, file=None, batch_size=10, reduce_num=0, test_part_start=0.9, test_part_end=1):
        if file is None:
            return

        data_temp = pd.read_csv(file)
        data_temp["predictions"] = data_temp["predictions"].apply(eval)

        x_temp = data_temp["predictions"].tolist()[reduce_num:]
        y_temp = data_temp["rate"].tolist()[reduce_num:]
        date = data_temp["date"].tolist()[reduce_num:]
        clas = data_temp["rate"].apply(lambda x: cls.classify(x)).tolist()[reduce_num:]

        data = []
        for i in range(len(x_temp)):
            data.append([x_temp[i], clas[i], date[i], y_temp[i]])

        data_length = len(data)
        train_length_1 = 0
        train_length_2 = int(data_length * test_part_start)
        train_length_3 = int(data_length * test_part_end)
        train_length_4 = data_length
        
        train_data = data[train_length_1:train_length_2] + data[train_length_3:train_length_4]
        test_data = data[train_length_2:train_length_3]
        random.shuffle(train_data)

        batch_index = []
        x_train = []
        y_train = []
        train_length = len(train_data)
        for i in range(train_length):
            x_train.append(train_data[i][0])
            y_train.append(train_data[i][1])
            if i % batch_size==0:
               batch_index.append(i)
        batch_index.append(train_length)

        x_test = []
        y_test = []
        test_others = []
        for i in range(len(test_data)):
            x_test.append(test_data[i][0])
            y_test.append(test_data[i][1])
            test_others.append([test_data[i][3], test_data[i][2]])

        return np.array(x_train), np.array(y_train), np.array(batch_index), np.array(x_test), np.array(y_test), np.array(test_others)    

    def train(self, basic_path=None, input_file=None, model_folder=None, folder_extra='', batch_size=30, reduce_num=0, test_part_start=0.9, test_part_end=1, times=10):
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if input_file is None or model_folder is None:
            return None

        model_path = os.path.join(basic_path, model_folder, "checkpoints" + folder_extra)
        input_file = os.path.join(model_path, input_file)
        
        x_train, y_train, batch_index, _, _, _ = self.get_data(file=input_file, batch_size=batch_size, reduce_num=0, test_part_start=test_part_start, test_part_end=test_part_end)

        tf.reset_default_graph()
        sl = SimpleLearn(
            input_size = x_train.shape[1],
            num_classes = y_train.shape[1],
            l2_reg_lambda = 0.01,
            learn_rate = self.learn_rate)

        checkpoint_prefix = os.path.join(model_path, "model")
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(times):
            for step in range(len(batch_index)-1):
                feed_dict = {
                    sl.input_x: x_train[batch_index[step]:batch_index[step+1]],
                    sl.input_y: y_train[batch_index[step]:batch_index[step+1]],
                    sl.dropout_keep_prob: self.drop
                }

                _, loss, global_step, accuracy, predictions, input_y_index = sess.run(
                    [sl.train_op, sl.global_step, sl.loss, sl.accuracy, sl.predictions, sl.input_y_index],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {:g}, loss {:g}, acc {:g}".format(time_str, global_step, loss, accuracy))
                
                all_accuracy = sl.various_accuracy(y_train.shape[1], input_y_index.tolist(), predictions.tolist())
                for a in all_accuracy:
                    print("input_nums: {:g}, pre_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(a[0], a[1], a[2], a[3]))                

            if i % 5 == 0:
                print("保存模型：", saver.save(sess, checkpoint_prefix))
        print("保存模型：", saver.save(sess, checkpoint_prefix))
        print("The train has finished")

    def predict(self, basic_path=None, input_file=None, model_folder=None, folder_extra='', reduce_num=0, test_part_start=0.9, test_part_end=1):
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if input_file is None or model_folder is None:
            return None

        model_path = os.path.join(basic_path, model_folder, "checkpoints" + folder_extra)
        input_file = os.path.join(model_path, input_file)
        
        _, _, _, x_test, y_test, y_others = self.get_data(file=input_file, batch_size=10, reduce_num=0, test_part_start=test_part_start, test_part_end=test_part_end)        

        print("x.shape = {}".format(x_test.shape))
        print("y.shape = {}".format(y_test.shape))

        tf.reset_default_graph()
        sl = SimpleLearn(
            input_size = x_test.shape[1],
            num_classes = y_test.shape[1],
            l2_reg_lambda = 0.01,
            learn_rate = self.learn_rate)

        checkpoint_prefix = os.path.join(model_path, "model")

        saver = tf.train.Saver()
        with tf.Session() as sess:
            #参数恢复            
            saver.restore(sess, checkpoint_prefix)
            
            feed_dict = {
                sl.input_x: x_test,
                sl.input_y: y_test,
                sl.dropout_keep_prob: 1.0
            }

            loss, accuracy, all_predictions, all_input = sess.run(
                [sl.loss, sl.accuracy, sl.predictions, sl.input_y_index],
                feed_dict)

            all_accuracy = sl.various_accuracy(y_test.shape[1], all_input, all_predictions)
            for a in all_accuracy:
                print("input_nums: {:g}, pre_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(a[0], a[1], a[2], a[3]))

        profit, origin_profit = DataHelper.rateCalc(all_predictions, y_others, y_test.shape[1]-1)
        return accuracy, profit, origin_profit, all_predictions, y_others

    def predictAndMake(self, basic_path=None, input_file=None, model_folder=None, folder_extra='', output_file=None):
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if input_file is None or model_folder is None or output_file is None:
            return None

        model_path = os.path.join(basic_path, model_folder, "checkpoints" + folder_extra)
        input_file = os.path.join(model_path, input_file)
        data_file = os.path.join(model_path, output_file)
        
        _, _, _, x_test, y_test, y_others = self.get_data(file=input_file, batch_size=10, reduce_num=0, test_part_start=0, test_part_end=1)

        print("x.shape = {}".format(x_test.shape))
        print("y.shape = {}".format(y_test.shape))

        tf.reset_default_graph()
        sl = SimpleLearn(
            input_size = x_test.shape[1],
            num_classes = y_test.shape[1],
            l2_reg_lambda = 0.01,
            learn_rate = 0.004)

        checkpoint_prefix = os.path.join(model_path, "model")

        saver = tf.train.Saver()
        with tf.Session() as sess:
            #参数恢复            
            saver.restore(sess, checkpoint_prefix)
            
            feed_dict = {
                sl.input_x: x_test,
                sl.input_y: y_test,
                sl.dropout_keep_prob: 1.0
            }

            loss, accuracy, all_predictions, all_input = sess.run(
                [sl.loss, sl.accuracy, sl.predictions, sl.input_y_index],
                feed_dict)

        data = {"date": [], "rate": [], "predictions": []}
        for k in range(len(x_test)):            
            data["date"].append(y_others[k][1])
            data["rate"].append(y_others[k][0])
            data["predictions"].append(x_test[k].tolist() + [all_predictions[k]])
        pd.DataFrame(data).to_csv(data_file, index=False, columns=["date", "rate", "predictions"])


class SimpleLearn(object):
    def __init__(self, input_size, num_classes, learn_rate, l2_reg_lambda=0.0):

        # Placeholders for input, output, dropout
        self.input_x = tf.placeholder(tf.float32, [None, input_size], name = "input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.01) # 0.0

        # Add dropout
        with tf.name_scope("dropout"):
            self.drop = tf.nn.dropout(self.input_x, self.dropout_keep_prob)

        W = self.weight_variable([input_size, num_classes])
        b = self.bias_variable([num_classes])
        
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.matmul(self.drop, W) + b
        
        self.predictions = tf.argmax(self.scores, 1, name = "predictions")
        self.input_y_index = tf.argmax(self.input_y, 1, name = "input_y_index")

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)) + l2_reg_lambda * l2_loss

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(self.loss, global_step=self.global_step)
        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y_index)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")

    def weight_variable(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def various_accuracy(self, num_labels=None, y_input_index=None, y_pre_index=None):
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