#coding=utf-8
import pymysql
import requests
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import random

from tensorflow.contrib import rnn
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Spider.tool import VTool

class LSTMStockOrigin():
    left_shift = 2

    @classmethod
    def classify(cls, y):
        y = float(y)
        if y <= 0.0:
            return [1,0]
        else:
            return [0,1]

    def various_accuracy(self, num_labels=None, y_input_index=None, y_pre_index=None):
        if num_labels == None or y_input_index == None or y_pre_index == None:
            return None

        #输入次数 预测次数(含对和错) 正确次数 正确率
        all_right_num = 0
        accuracys = []
        for i in range(num_labels):
            accuracys.append([0,0,0,0])
        for i in range(len(y_input_index)):
            accuracys[y_input_index[i]][0] += 1
            accuracys[y_pre_index[i]][1] += 1
            if y_input_index[i] == y_pre_index[i]:
                accuracys[y_input_index[i]][2] += 1
                all_right_num += 1

        for i in range(len(accuracys)):
            if accuracys[i][0] != 0:
                accuracys[i][3] = float(accuracys[i][2]) / accuracys[i][0]

        return len(y_input_index), all_right_num, accuracys

    #获取训练集
    @classmethod
    def get_train_data(cls, data=None, save_path=None, input_size=7, batch_size=30, time_step=10, reduce_num=0, test_part_start=0.9, test_part_end=1, configure={}):
        left_shift = 2
        for i in range(len(data)-left_shift):
            data[i][input_size] = data[i+left_shift][input_size]
        data = data[:-left_shift]
        data = data[reduce_num:]

        save_path = os.path.join(save_path, "temp.csv")
        con = {}
        for k in configure:
            con[k] = [configure[k]]
        pd.DataFrame(con).to_csv(save_path, index=False)

        y_classify = []
        for d in data:
            y_classify.append(cls.classify(d[input_size]))
        
        x_all = []
        y_all = []
        for i in range(len(data)-time_step):
            x = data[i:i+time_step,:input_size]
            y = y_classify[i:i+time_step]
            x_all.append(x.tolist())
            y_all.append(y)
        
        data_length = len(x_all)
        train_length_1 = 0
        train_length_2 = int(data_length * test_part_start)
        train_length_3 = int(data_length * test_part_end)
        train_length_3 = train_length_3 + time_step - 1 if train_length_3 + time_step - 1 < data_length else train_length_3 #防止测试集数据被使用
        train_length_4 = data_length
        
        x_temp = x_all[train_length_1: train_length_2] + x_all[train_length_3: train_length_4]
        y_train = y_all[train_length_1: train_length_2] + y_all[train_length_3: train_length_4]
        
        x_train = []        
        for x in x_temp:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            x_train.append((x - mean) / std) #标准化

        index = []
        for i in range(len(x_train)):
            index.append(i)
        random.shuffle(index)
        x_random = []
        y_random = []
        for i in range(len(index)):
            x_random.append(x_train[i])
            y_random.append(y_train[i])

        return x_random, y_random


    #获取测试集
    @classmethod
    def get_test_data(cls, data=None, data_others=None, save_path=None, input_size=7, time_step=10, reduce_num=0, test_part_start=0.9, test_part_end=1):
        left_shift = 2
        for i in range(len(data)-left_shift):
            data[i][input_size] = data[i+left_shift][input_size]
            data_others[i] = data_others[i+left_shift]
        data = data[:-left_shift]
        data = data[reduce_num:]
        data_others = data_others[:-left_shift]
        data_others = data_others[reduce_num:]

        y_classify = []
        for d in data:
            y_classify.append(cls.classify(d[input_size]))
        
        x_all = []
        y_all = []
        for i in range(len(data)-time_step):
            x = data[i:i+time_step,:input_size]
            y = y_classify[i:i+time_step]
            x_all.append(x.tolist())
            y_all.append(y)
        
        data_length = len(x_all)
        train_length_1 = int(data_length * test_part_start)
        train_length_2 = int(data_length * test_part_end)
        
        x_temp = x_all[train_length_1: train_length_2]
        y_temp = y_all[train_length_1: train_length_2]
        y_others = data_others[time_step-1:-1]
        y_others = y_others[train_length_1: train_length_2]

        x_test = []        
        for x in x_temp:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            x_test.append((x - mean) / std) #标准化

        y_test = []
        for i in range(len(y_temp)):
            y_test.append(y_temp[i][-1])

        return x_test, y_test, y_others

    #——————————————————导入数据——————————————————————    
    def lstm(self, X, keep_prob, rnn_unit=10, input_size=7, output_size=1):

        #输入层、输出层权重、偏置
        weights = {
            'in':tf.Variable(tf.random_normal([input_size, rnn_unit])),
            'out':tf.Variable(tf.random_normal([rnn_unit, output_size]))
        }
        biases = {
            'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
            'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
        }
        
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        
        # input layer
        w_in = weights['in']
        b_in = biases['in']  
        input = tf.reshape(X,[-1,input_size])
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  #将tensor转成3维，作为lstm cell的输入
       
        def LstmCell():
            #定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
            lstm_cell = rnn.BasicLSTMCell(num_units=rnn_unit, forget_bias=1.0, state_is_tuple=True)
            #添加 dropout layer, 一般只设置 output_keep_prob
            lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            return lstm_cell

        #调用 MultiRNNCell 来实现多层 LSTM
        mlstm_cell = rnn.MultiRNNCell([LstmCell() for _ in range(2)], state_is_tuple=True)
        #用全零来初始化state
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

        # cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        # init_state = cell.zero_state(batch_size,dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(mlstm_cell, input_rnn, initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        
        # softmax output layer
        output = tf.reshape(output_rnn, [-1, rnn_unit])
        w_out = weights['out']
        b_out = biases['out']
        # tf.nn.softmax()
        pred = tf.matmul(output, w_out) + b_out
        predictions = tf.argmax(pred, 1, name = "predictions")        

        return pred, predictions, final_states

    def getColumns(self, input_type="", word_count=0, res_type=""):
        columns = []
        if input_type == "news":
            columns = ["news_pos_num", "news_neg_num", "opening", "difference", "percentage_difference", "lowest", "highest", "volume", "amount", "closing"]
        elif input_type == "bindex":
            word_columns = []
            for i in range(word_count):
                word_columns.append("word%s" % (i+1))
            columns = word_columns + ["opening", "difference", "percentage_difference", "lowest", "highest", "volume", "amount", "closing"]
        else:
            columns = ["opening", "difference", "percentage_difference", "lowest", "highest", "volume", "amount", "closing"]

        if res_type != "":
            columns.append(res_type)

        return columns    

    #——————————————————训练模型——————————————————
    def train_rate(self, basic_path=None, data_file=None, model_folder=None, folder_extra="", input_type="origin", word_count=0, input_size=8, batch_size=30, time_step=10, reduce_num=0, test_part_start=0.9, test_part_end=1, times=50):
        if data_file is None or model_folder is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(basic_path, data_file)
        model_path = os.path.join(basic_path, model_folder)        
        VTool.makeDirs(folders=[model_path])
        
        f = open(data_path)
        df = pd.read_csv(f)
        f.close()
        columns = self.getColumns(input_type=input_type, word_count=word_count, res_type="rate")
        data = df[columns].values

        rnn_unit = 8       #单元数量        
        output_size = 2        

        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=[None,time_step,input_size+word_count])
        Y = tf.placeholder(tf.float32, shape=[None,time_step,output_size])
        keep_prob = tf.placeholder(tf.float32)
        
        x_train, y_train=self.get_train_data(data, model_path, input_size+word_count, batch_size, time_step, reduce_num, test_part_start, test_part_end, {"input_type":input_type, "word_count":word_count, "input_size":input_size, "time_step":time_step, "rnn_unit":rnn_unit, "output_size":output_size})
        
        pred, predictions, _ = self.lstm(X=X, keep_prob=keep_prob, rnn_unit=rnn_unit, input_size=input_size+word_count, output_size=output_size)
        
        global_step = tf.Variable(0)
        lr = 0.01
        #learning_rate = tf.train.exponential_decay(0.01, global_step, decay_steps=len(x_train), decay_rate=0.95, staircase=True)        

        #损失函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = tf.reshape(Y,[-1,2])))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) 
        saver = tf.train.Saver()

        y_input = tf.argmax(tf.reshape(Y,[-1,2]), 1)
        correct_predictions = tf.equal(predictions, y_input)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        checkpoint_dir = os.path.abspath(os.path.join(model_path, "checkpoints"+folder_extra))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(times):
                j = 0
                while j < len(x_train):
                    j_end = j + batch_size if j + batch_size < len(x_train) else len(x_train)
                    _, _loss = sess.run([train_op, loss], feed_dict={X:x_train[j:j_end], Y:y_train[j:j_end], keep_prob:0.8})
                    j = j_end
                print("Number of iterations:", i, " loss:", _loss)

                if i % 10 == 0:
                    print("保存模型：",saver.save(sess, checkpoint_prefix))

                    # _predictions = sess.run([predictions],feed_dict={X:x_test, keep_prob:1})
                    # _predictions = np.array(_predictions).reshape((-1, time_step)).tolist()
                    # y_predict = []
                    # for p in _predictions:
                    #     y_predict.append(p[-1])
                    # all_num, right_num, all_accuracy = self.various_accuracy(output_size, y_test, y_predict)
                    # print("All input_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(all_num, right_num, right_num/all_num))
                    # for a in all_accuracy:
                    #     print("input_nums: {:g}, pre_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(a[0], a[1], a[2], a[3]))

            print("保存模型：",saver.save(sess, checkpoint_prefix))
            print("The train has finished")

    #——————————————————预测模型——————————————————
    def predict_rate(self, basic_path=None, data_file=None, model_folder=None, folder_extra="", reduce_num=0, test_part_start=0.9, test_part_end=1):
        if data_file is None or model_folder is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(basic_path, data_file)
        model_path = os.path.join(basic_path, model_folder)

        save_path = os.path.join(model_path, "temp.csv")
        t = open(save_path)
        dt = pd.read_csv(t)
        t.close()
        configure = {}
        for k in dt:
            configure[k] = dt[k].values[-1]

        f = open(data_path)
        df = pd.read_csv(f)
        f.close()
        columns = self.getColumns(input_type=configure["input_type"], word_count=configure["word_count"], res_type="rate")
        data = df[columns].values
        data_others = df[["rate", "date"]].values

        rnn_unit = configure["rnn_unit"]
        input_size = configure["input_size"]+configure["word_count"]
        time_step = configure["time_step"]
        output_size = configure["output_size"]
        
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=[None,time_step,input_size])
        keep_prob = tf.placeholder(tf.float32)
        
        x_test, y_test, y_others = self.get_test_data(data, data_others, model_path, input_size, time_step, reduce_num, test_part_start, test_part_end)        
        
        pred, predictions, _ = self.lstm(X=X, keep_prob=keep_prob, rnn_unit=rnn_unit, input_size=input_size, output_size=output_size)
        saver = tf.train.Saver()

        for i in range(len(y_test)):
            y_test[i] = np.argmax(y_test[i])

        with tf.Session() as sess:
            #参数恢复
            module_file = tf.train.latest_checkpoint(os.path.join(model_path, "checkpoints"+folder_extra))
            saver.restore(sess, module_file)             
            
            _predictions = sess.run([predictions],feed_dict={X:x_test, keep_prob:1})
            _predictions = np.array(_predictions).reshape((-1, time_step)).tolist()
            y_predict = []
            for p in _predictions:
                y_predict.append(p[-1])
            all_num, right_num, all_accuracy = self.various_accuracy(output_size, y_test, y_predict)
            accuracy = right_num/all_num
            print("All input_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(all_num, right_num, accuracy))
            for a in all_accuracy:
                print("input_nums: {:g}, pre_nums: {:g}, right_nums: {:g}, accuracy: {:g}".format(a[0], a[1], a[2], a[3]))

        profit, origin_profit = self.rateCalc(y_predict, y_others)
        return accuracy, profit, origin_profit, y_predict, y_others

    @classmethod
    def rateCalc(cls, test_predict, test_others, buy_when=1, start_date=None, end_date=None):
        sindex = 0
        eindex = len(test_predict)

        if start_date != None:
            for i in range(len(test_others)):
                if test_others[i][1] >= start_date:
                    sindex = i
                    break

        if end_date != None:
            for ii in range(i, len(test_others)):
                if test_others[ii][1] >= end_date:
                    eindex = ii
                    break
        
        sum_rate = 0
        sum_all_rate = 0
        if (eindex > 0):
            for i in range(sindex, eindex):
                if test_predict[i] == buy_when:
                    sum_rate = sum_rate + float(test_others[i][0]) * 1.0
                sum_all_rate = sum_all_rate + float(test_others[i][0]) * 1.0
            print( "\n%s 到 %s，总收益率为：%f，一直购买收益率为：%f，总交易天数为：%d" % (test_others[sindex][1], test_others[eindex-1][1], sum_rate, sum_all_rate, eindex-sindex) )
        return sum_rate, sum_all_rate