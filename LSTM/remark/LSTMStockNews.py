#coding=utf-8
import pymysql
import requests
import json
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

from tensorflow.contrib import rnn
from pandas.core.frame import DataFrame

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Spider.tool import VTool

class LSTMStockNews():    

    def __init__(self,):
        self.file_paths = {}
        self.batch_size = 0
        self.read_size = 0
        self.train_size = 0
        self.test_size = 0

    def init(self,):
        for path in self.file_paths:        
            self.file_paths[path].close()
        
        self.file_paths = {}
        self.batch_size = 0
        self.read_size = 0
        self.train_size = 0
        self.test_size = 0
    
    @classmethod
    def make_train_csv(cls, cur=None, start_date=None, end_date=None, basic_path=None, word_trend_file=None, news_file=None, output_file=None, time_step=3, stock_id_str=None):
        if cur == None or start_date == None or end_date == None or word_trend_file is None or output_file == None or stock_id_str == None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if time_step < 0:
            time_step = 3
        news_path = os.path.join(basic_path, news_file)
        word_trend_path = os.path.join(basic_path, word_trend_file)
        output_path = os.path.join(basic_path, output_file)
        VTool.makeDirs(files=[output_path])
        pd.DataFrame({"0":[], "1":[]}).to_csv(output_path, index=False)

        word_trend = {}
        word_trend_temp = pd.read_csv(word_trend_path)
        for k in word_trend_temp["0"].keys():
            word_trend[word_trend_temp["0"][k]] = [word_trend_temp["1"][k], word_trend_temp["2"][k]]
        p_up = word_trend['total_words'][0] / (word_trend['total_words'][0] + word_trend['total_words'][1])
        p_down = word_trend['total_words'][1] / (word_trend['total_words'][0] + word_trend['total_words'][1])

        cur.execute("SELECT count(*) as count FROM history WHERE stock_id in (%s) and date between '%s' and '%s' " % (stock_id_str, start_date, end_date))
        count = cur.fetchall()
        count = count[0][0]	    
        stock_id_num = len(stock_id_str.split(","))
        skip = 50 * stock_id_num
        slimit = 0
        while slimit < count:
            cur.execute("SELECT stock_id, opening, closing, difference, percentage_difference, lowest, highest, volume, amount, date FROM history WHERE stock_id in (%s) and date between '%s' and '%s' order by date asc, stock_id asc limit %d,%d " % (stock_id_str, start_date, end_date, 0 if slimit-stock_id_num < 0 else slimit-stock_id_num, skip if slimit-stock_id_num < 0 else skip+stock_id_num))
            slimit += skip
            history_tt = cur.fetchall()
            history_t = []
            for h in history_tt:
                history_t.append([int(h[0]), float(h[1]), float(h[2]), float(h[3]), float(h[4]), float(h[5]), float(h[6]), float(h[7]), float(h[8]), str(h[9])])
            del history_tt

            history_temp = []
            for h in zip(*history_t):
                history_temp.append(h)
            history = {'stock_id':history_temp[0], 'opening':history_temp[1], 'closing':history_temp[2], 'difference':history_temp[3], 'percentage_difference':history_temp[4], 'lowest':history_temp[5], 'highest':history_temp[6], 'volume':history_temp[7], 'amount':history_temp[8], 'date':history_temp[9]}
            del history_t, history_temp        
            history = DataFrame(history)
            g_history = history.groupby(by = ['stock_id'])
            #0.01 -> 1 % 保留2位小数
            history['rate'] = 100 * (g_history.shift(0)["closing"] / g_history.shift(1)["closing"] - 1)
            history.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
            '''
            '''
            sdate = str(history['date'][history['date'].keys()[0]])
            edate = str(history['date'][history['date'].keys()[-1]])
            sdate = datetime.datetime.strptime(sdate,'%Y-%m-%d')
            sdate = (sdate - datetime.timedelta(days=time_step)).strftime('%Y-%m-%d')
            cur.execute("SELECT GROUP_CONCAT(id  SEPARATOR ','), time FROM news WHERE time between '%s' and '%s' group by time" % (sdate, edate))            
            news_temp = cur.fetchall()        
            news_by_date = {}
            news_by_id = {}
            for n in news_temp:
                news_by_date[str(n[1])] = n[0].split(",")
                for nid in news_by_date[str(n[1])]:
                    news_by_id[nid] = None
            del news_temp        

            nid_len = len(news_by_id)
            reader = pd.read_csv(news_path, chunksize=1000)
            for sentences in reader:
                if nid_len > 0:
                    for k in sentences['1'].keys():
                        nid = str(sentences['0'][k])
                        if nid in news_by_id and news_by_id[nid] == None:
                            news_by_id[nid] = str(sentences['1'][k]).split(" ")
                            wp_up = p_up
                            wp_down = p_down
                            for w in news_by_id[nid]:
                                if w not in word_trend:
                                    wp_up *= (1 / word_trend['total_words'][0])
                                    wp_down *= (1 / word_trend['total_words'][1])
                                else:
                                    if word_trend[w][0] > 0:
                                        wp_up *= word_trend[w][0]
                                    else:
                                        wp_up *= (1 / word_trend['total_words'][0])

                                    if word_trend[w][1] > 0:
                                        wp_down *= word_trend[w][1]
                                    else:
                                        wp_down *= (1 / word_trend['total_words'][1])
                                while True:
                                    if wp_up < 1 and wp_down < 1:
                                        wp_up *= 10
                                        wp_down *= 10
                                    else:
                                        break

                            news_by_id[nid] = [wp_up / (wp_up + wp_down), -1*wp_down / (wp_up + wp_down)]	                    
                            nid_len-=1                            
                            if nid_len <= 0:
                                break
                else:
                    break
            reader.close()
            del reader, sentences

            for d in news_by_date:
                sumn = [0, 0]
                for nid in news_by_date[d]:
                    sumn[0] += news_by_id[nid][0]
                    sumn[1] += news_by_id[nid][1]
                le = len(news_by_date[d])
                if le > 0:
                    sumn[0] /= le
                    sumn[1] /= le
                news_by_date[d] = sumn
                print(d)

            history['news_pos_num'] = 0
            history['news_neg_num'] = 0
            for i in history.index:                
                history.loc[i, 'rate'] = str(np.round(float(history['rate'][i]), 2))
                if str(history['date'][i]) in news_by_date:
                    history.loc[i, 'news_pos_num'] = str(np.round(float(news_by_date[str(history['date'][i])][0]), 2))
                    history.loc[i, 'news_neg_num'] = str(np.round(float(news_by_date[str(history['date'][i])][1]), 2))
                else:
                    history.loc[i, 'news_pos_num'] = "0"
                    history.loc[i, 'news_neg_num'] = "0"
            
            #将经过标准化的数据处理成训练集和测试集可接受的形式              
            def func_train_data(data_stock, time_step):                
                if cls.groupby_skip == False:
                    cls.groupby_skip = True
                    return None
                print ("正在处理的股票代码:%06s"%data_stock.name)            
                #提取输入S列（对应train_x）
                data_temp_x = data_stock[["news_pos_num", "news_neg_num", "opening", "closing", "difference", "percentage_difference", "lowest", "highest", "volume", "amount"]]
                #提取输出列（对应train_y）
                data_temp_y = data_stock[["rate", "date", "stock_id"]]
                data_res = []
                for i in range(time_step - 1, len(data_temp_x.index) - 1):               
                    data_res.append( data_temp_x.iloc[i - time_step + 1: i + 1].values.reshape(1, time_step * 10).tolist() + data_temp_y.iloc[i + 1].values.reshape(1,3).tolist() )
                if len(data_res) != 0:
                    pd.DataFrame(data_res).to_csv(output_path, index=False, header=False, mode="a")
            
            g_stock = history.groupby(by = ["stock_id"])
            #清空接收路径下的文件，初始化列名	                    
            cls.groupby_skip = False
            g_stock.apply(func_train_data, time_step = time_step)

    def get_train_data(self, file_path, time_step=10, rtype="train"):
        """获取训练集和测试集数据"""
        self_variable_name = "read_{0}".format(file_path)
        if self_variable_name in self.file_paths:
            cursor = self.file_paths[self_variable_name]
        else:
            self.file_paths[self_variable_name] = cursor = pd.read_csv(file_path, iterator = True)

        size = self.batch_size
        if rtype == "train":
            if self.read_size >= self.train_size:
                return None, None                
        elif rtype == "test":
            size = self.test_size
        else:
            return None, None
                
        data_temp = cursor.get_chunk(size)
        self.read_size += size
        data_temp['0'] = data_temp['0'].apply(eval)        
        data_temp['1'] = data_temp['1'].apply(eval)
        data_temp["1"] = data_temp["1"].apply(lambda x: x[0])

        #处理成矩阵形式返回
        return np.array(data_temp["0"].tolist()).reshape(size , time_step * (8+2)), np.array(data_temp["1"].tolist()).reshape(size , 1)
    
    def get_train_softmax(self, file_path, time_step=10, rtype="train"):
        """
        将收益变成分类问题。分类定义在classify函数内，分别为：
            0: <= -0.2%
            1: [-0.2%, 0.2%)
            2: [0.2%, 以上)
        """

        self_variable_name = "read_{0}".format(file_path)
        if self_variable_name in self.file_paths:
            cursor = self.file_paths[self_variable_name]
        else:
            self.file_paths[self_variable_name] = cursor = pd.read_csv(file_path, iterator = True)
        
        size = self.batch_size
        if rtype == "train":
            if self.read_size >= self.train_size:
                return None, None
        elif rtype == "test":
            size = self.test_size
        else:
            return None, None
        
        data_temp = cursor.get_chunk(size)
        self.read_size += size
        data_temp['0'] = data_temp['0'].apply(eval)
        data_temp['1'] = data_temp['1'].apply(eval)
        data_temp["1"] = data_temp["1"].apply(lambda x: x[0])

        def classify(y):
            y = float(y)
            if y < -0.2:
                return [1,0,0]
            elif y > 0.2:
                return [0,0,1]
            else:
                return [0,1,0]

        data_temp["1"] = data_temp["1"].apply(lambda x: classify(x))
        return np.array(data_temp["0"].tolist()).reshape(size , time_step * (8+2)), np.array(data_temp["1"].tolist()).reshape(size , 3)

    def stock_lstm(self, basic_path=None, train_file=None, model_file=None, log_folder=None, pre_file=None):
        """
        使用LSTM处理股票数据
        直接预测收益
        """
        if train_file is None or model_file is None or log_folder is None or pre_file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(basic_path, train_file)
        model_path = os.path.join(basic_path, model_file)
        log_path = os.path.join(basic_path, log_folder)
        pre_path = os.path.join(basic_path, pre_file)
        VTool.makeDirs(files=[model_path, pre_path], folders=[log_path])        

        tf.reset_default_graph()
        #给batch_size赋值
        self.batch_size = 20
        test_part = 0.1
        self.train_size, self.test_size = VTool.initCsvTrainAndTest(basic_path=basic_path, input_file=train_file, batch_size=self.batch_size, test_part=test_part)        
        #学习率
        learning_rate = 0.001
        #喂数据给LSTM的原始数据有几行，即：一次希望LSTM能“看到”多少个交易日的数据
        origin_data_row = 3
        #喂给LSTM的原始数据有几列，即：日线数据有几个元素
        origin_data_col = 8+2
        #LSTM网络有几层
        layer_num = 1
        #LSTM网络，每层有几个神经元
        cell_num = 128
        #最后输出的数据维度，即：要预测几个数据，该处只预测收益率，只有一个数据
        output_num = 1
        #每次给LSTM网络喂多少行经过处理的股票数据。该参数依据自己显卡和网络大小动态调整，越大一次处理的就越多，越能占用更多的计算资源
        batch_size = tf.placeholder(tf.int32, [])
        #输入层、输出层权重、偏置。
        #通过这两对参数，LSTM层能够匹配输入和输出的数据
        W = {
            'in':tf.Variable(tf.truncated_normal([origin_data_col, cell_num], stddev = 1), dtype = tf.float32),
            'out':tf.Variable(tf.truncated_normal([cell_num, output_num], stddev = 1), dtype = tf.float32)
        }
        bias = {
            'in':tf.Variable(tf.constant(0.1, shape=[cell_num,]), dtype = tf.float32),
            'out':tf.Variable(tf.constant(0.1, shape=[output_num,]), dtype = tf.float32)
        }
        #告诉LSTM网络，即将要喂的数据是几行几列
        #None的意思就是喂数据时，行数不确定交给tf自动匹配
        #我们喂得数据行数其实就是batch_size，但是因为None这个位置tf只接受数字变量，而batch_size是placeholder定义的Tensor变量，表示我们在喂数据的时候才会告诉tf具体的值是多少
        input_x = tf.placeholder(tf.float32, [None, origin_data_col * origin_data_row])
        input_y = tf.placeholder(tf.float32, [None, output_num])
        #处理过拟合问题。该值在其起作用的层上，给该层每一个神经元添加一个“开关”，“开关”打开的概率是keep_prob定义的值，一旦开关被关了，这个神经元的输出将被“阻断”。这样做可以平衡各个神经元起作用的重要性，杜绝某一个神经元“一家独大”，各种大佬都证明这种方法可以有效减弱过拟合的风险。
        keep_prob = tf.placeholder(tf.float32, [])

        #通过reshape将输入的input_x转化成2维，-1表示函数自己判断该是多少行，列必须是origin_data_col
        #转化成2维 是因为即将要做矩阵乘法，矩阵一般都是2维的（反正我没见过3维的）
        input_x_after_reshape_2 = tf.reshape(input_x, [-1, origin_data_col])

        #当前计算的这一行，就是输入层。输入层的激活函数是relu,并且施加一个“开关”，其打开的概率为keep_prob
        #input_rnn即是输入层的输出，也是下一层--LSTM层的输入
        input_rnn = tf.nn.dropout(tf.nn.relu_layer(input_x_after_reshape_2, W['in'], bias['in']), keep_prob)
        
        #通过reshape将输入的input_rnn转化成3维
        #转化成3维，是因为即将要进入LSTM层，接收3个维度的数据。粗糙点说，即LSTM接受：batch_size个，origin_data_row行cell_num列的矩阵，这里写-1的原因与input_x写None一致
        input_rnn = tf.reshape(input_rnn, [-1, origin_data_row, cell_num])

        #定义一个带着“开关”的LSTM单层，一般管它叫细胞
        def lstm_cell():
            cell = rnn.LSTMCell(cell_num, reuse = tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(cell, output_keep_prob = keep_prob)
        #这一行就是tensorflow定义多层LSTM网络的代码
        lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple = True)
        #初始化LSTM网络
        init_state = lstm_layers.zero_state(batch_size, dtype = tf.float32)

        #使用dynamic_rnn函数，告知tf构建多层LSTM网络，并定义该层的输出
        outputs, state = tf.nn.dynamic_rnn(lstm_layers, inputs = input_rnn, initial_state = init_state, time_major = False)        
        h_state = state[-1][1]

        #该行代码表示了输出层
        #将LSTM层的输出，输入到输出层，输出层最终得出的值即为预测的收益
        y_pre = tf.matmul(h_state, W['out']) + bias['out']

        #损失函数，用作指导tf
        #loss的定义为：(使用预测的收益 - 喂给LSTM的这60个交易日对应的真实收益)的平方，如果有多个（即：batch_size个且batch_size大于1），那就求一下平均值（先挨个求平方，再整个求平均值）
        loss = tf.reduce_mean(tf.square(tf.subtract(y_pre, input_y)))
        #告诉tf，它需要做的事情就是就是尽可能将loss减小
        #learning_rate是减小的这个过程中的参数。如果将我们的目标比喻为“从北向南走路走到菜市场”，我理解的是
        #learning_rate越大，我们走的每一步就迈的越大。初看似乎步子越大越好，但是其实并不能保证每一步都是向南走
        #的，有可能因为训练数据的原因，导致我们朝西走了一大步。或者我们马上就要到菜市场了，但是一大步走过去，给
        #走过了。。。综上，这个learning_rate（学习率参数）的取值，无法给出一个比较普适的，还是需要根据实际情况去
        #尝试和调整。0.001的取值是tf给的默认值
        #上述例子是个人理解用尽可能通俗易懂地语言表达。如有错误，欢迎指正
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        #这块定义了一个新的值，用作展示训练的效果
        #它的定义为：选择预测值和实际值差别最大的情况并将差值返回
        accuracy = tf.reduce_max(tf.abs(tf.subtract(y_pre, input_y)))
                
        #设置tf按需使用GPU资源，而不是有多少就占用多少
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        sess = tf.Session(config = config)  
        merged = tf.summary.merge_all()  
        # save the logs  
        writer = tf.summary.FileWriter(log_path, sess.graph)  

        #tf要求必须如此定义一个init变量，用以在初始化运行（也就是没有保存模型）时加载各个变量
        init = tf.global_variables_initializer()        
        #用以保存参数的函数（跑完下次再跑，就可以直接读取上次跑的结果而不必重头开始）
        saver = tf.train.Saver(tf.global_variables())

        #使用with，保证执行完后正常关闭tf
        with sess and open (pre_path, "w") as f:
            try:
                #定义了存储模型的文件路径，即：当前运行的python文件路径下，文件名为stock_rnn.ckpt
                saver.restore(sess, model_path)
                print ("成功加载模型参数")
            except:
                #如果是第一次运行，通过init告知tf加载并初始化变量
                print ("未加载模型参数，文件被删除或者第一次运行")
                sess.run(init)
                       
            i = 0  
            while True:
                #读取训练集数据
                train_x, train_y = self.get_train_data(file_path = train_path, time_step = origin_data_row, rtype="train")
                if train_x is None:
                    print ("训练集均已训练完毕")                    
                    saver.save(sess, model_path)
                    print("保存模型\n")
                    break

                if (i + 1) % 10 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        input_x:train_x, input_y: train_y, keep_prob: 1.0, batch_size: self.batch_size})
                    #输出
                    print ("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
                    saver.save(sess, model_path)
                    print("保存模型\n")
                    ############################################
                    result = sess.run(merged, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size:  self.batch_size})  
                    writer.add_summary(result, i)

                    #这部分代码作用为：每次保存模型，顺便将预测收益和真实收益输出保存至show_y_pre.txt文件下。熟悉tf可视化，完全可以采用可视化替代
                    _y_pre_train = sess.run(y_pre, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size:  self.batch_size})
                    _loss = sess.run(loss, feed_dict={
                        input_x:train_x, input_y: train_y, keep_prob: 1.0, batch_size: self.batch_size})
                    a1 = np.array(train_y).reshape(1, self.batch_size)                
                    b1 = np.array(_y_pre_train).reshape(1, self.batch_size)                    
                    
                    f.write(str(a1.tolist()))
                    f.write("\n")
                    f.write(str(b1.tolist()))
                    f.write("\n")
                    f.write(str(_loss))
                    f.write("\n")
                    ############################################
                i += 1
                #按照给定的参数训练一次LSTM神经网络
                sess.run(train_op, feed_dict={input_x: train_x, input_y: train_y, keep_prob: 0.6, batch_size: self.batch_size})

            #计算测试数据的准确率
            #读取测试集数据            
            test_x, test_y = self.get_train_data(file_path = train_path, time_step=origin_data_row, rtype="test")
            print ("test accuracy {0}".format(sess.run(accuracy, feed_dict={
                input_x: test_x, input_y: test_y, keep_prob: 1.0, batch_size:self.test_size})))
            self.init()
            

    def stock_lstm_softmax(self, basic_path=None, train_file=None, model_file=None, log_folder=None, pre_file=None):        
        """
        使用LSTM处理股票数据
        分类预测
        """
        if train_file is None or model_file is None or log_folder is None or pre_file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(basic_path, train_file)
        model_path = os.path.join(basic_path, model_file)
        log_path = os.path.join(basic_path, log_folder)
        pre_path = os.path.join(basic_path, pre_file)
        VTool.makeDirs(files=[model_path, pre_path], folders=[log_path])

        tf.reset_default_graph()
        #给batch_size赋值
        self.batch_size = 20
        test_part = 0.1
        self.train_size, self.test_size = VTool.initCsvTrainAndTest(basic_path=basic_path, input_file=train_file, batch_size=self.batch_size, test_part=test_part)
        #学习率
        learning_rate = 0.001
        #喂数据给LSTM的原始数据有几行，即：一次希望LSTM能“看到”多少个交易日的数据
        origin_data_row = 3
        #喂给LSTM的原始数据有几列，即：日线数据有几个元素
        origin_data_col = 8+2
        #LSTM网络有几层
        layer_num = 1
        #LSTM网络，每层有几个神经元
        cell_num = 256
        #最后输出的数据维度，即：要预测几个数据，该处需要处理分类问题，按照自己设定的类型数量设定
        output_num = 3
        #每次给LSTM网络喂多少行经过处理的股票数据。该参数依据自己显卡和网络大小动态调整，越大 一次处理的就越多，越能占用更多的计算资源
        batch_size = tf.placeholder(tf.int32, [])
        #输入层、输出层权重、偏置。
        #通过这两对参数，LSTM层能够匹配输入和输出的数据
        W = {
            'in':tf.Variable(tf.truncated_normal([origin_data_col, cell_num], stddev = 1), dtype = tf.float32),
            'out':tf.Variable(tf.truncated_normal([cell_num, output_num], stddev = 1), dtype = tf.float32)
        }
        bias = {
            'in':tf.Variable(tf.constant(0.1, shape=[cell_num,]), dtype = tf.float32),
            'out':tf.Variable(tf.constant(0.1, shape=[output_num,]), dtype = tf.float32)
        }
        #告诉LSTM网络，即将要喂的数据是几行几列
        #None的意思就是喂数据时，行数不确定交给tf自动匹配
        #我们喂得数据行数其实就是batch_size，但是因为None这个位置tf只接受数字变量，而batch_size是placeholder定义的Tensor变量，表示我们在喂数据的时候才会告诉tf具体的值是多少
        input_x = tf.placeholder(tf.float32, [None, origin_data_col * origin_data_row])
        input_y = tf.placeholder(tf.float32, [None, output_num])
        #处理过拟合问题。该值在其起作用的层上，给该层每一个神经元添加一个“开关”，“开关”打开的概率是keep_prob定义的值，一旦开关被关了，这个神经元的输出将被“阻断”。这样做可以平衡各个神经元起作用的重要性，杜绝某一个神经元“一家独大”，各种大佬都证明这种方法可以有效减弱过拟合的风险。
        keep_prob = tf.placeholder(tf.float32, [])

        #通过reshape将输入的input_x转化成2维，-1表示函数自己判断该是多少行，列必须是origin_data_col
        #转化成2维 是因为即将要做矩阵乘法，矩阵一般都是2维的（反正我没见过3维的）
        input_x_after_reshape_2 = tf.reshape(input_x, [-1, origin_data_col])

        #当前计算的这一行，就是输入层。输入层的激活函数是relu,并且施加一个“开关”，其打开的概率为keep_prob
        #input_rnn即是输入层的输出，也是下一层--LSTM层的输入
        input_rnn = tf.nn.dropout(tf.nn.relu_layer(input_x_after_reshape_2, W['in'], bias['in']), keep_prob)
        
        #通过reshape将输入的input_rnn转化成3维
        #转化成3维，是因为即将要进入LSTM层，接收3个维度的数据。粗糙点说，即LSTM接受：batch_size个，origin_data_row行cell_num列的矩阵，这里写-1的原因与input_x写None一致
        input_rnn = tf.reshape(input_rnn, [-1, origin_data_row, cell_num])

        #定义一个带着“开关”的LSTM单层，一般管它叫细胞
        def lstm_cell():
            cell = rnn.LSTMCell(cell_num, reuse=tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        #这一行就是tensorflow定义多层LSTM网络的代码
        lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple = True)
        #初始化LSTM网络
        init_state = lstm_layers.zero_state(batch_size, dtype = tf.float32)

        #使用dynamic_rnn函数，告知tf构建多层LSTM网络，并定义该层的输出
        outputs, state = tf.nn.dynamic_rnn(lstm_layers, inputs = input_rnn, initial_state = init_state, time_major = False)
        h_state = state[-1][1]

        #该行代码表示了输出层
        #将LSTM层的输出，输入到输出层（输出层带softmax激活函数），输出为各个分类的概率
        #假设有3个分类，那么输出举例为：[0.001, 0.992, 0.007]，表示第1种分类概率千分之1，第二种99.2%, 第三种千分之7
        y_pre = tf.nn.softmax(tf.matmul(h_state, W['out']) + bias['out'])

        #损失函数，用作指导tf
        #loss定义为交叉熵损失函数，softmax输出层大多都使用的这个损失函数。关于该损失函数详情可以百度下
        loss = -tf.reduce_mean(input_y * tf.log(y_pre))
        #告诉tf，它需要做的事情就是就是尽可能将loss减小
        #learning_rate是减小的这个过程中的参数。如果将我们的目标比喻为“从北向南走路走到菜市场”，我理解的是
        #learning_rate越大，我们走的每一步就迈的越大。初看似乎步子越大越好，但是其实并不能保证每一步都是向南走
        #的，有可能因为训练数据的原因，导致我们朝西走了一大步。或者我们马上就要到菜市场了，但是一大步走过去，给
        #走过了。。。综上，这个learning_rate（学习率参数）的取值，无法给出一个比较普适的，还是需要根据实际情况去
        #尝试和调整。0.001的取值是tf给的默认值
        #上述例子是个人理解用尽可能通俗易懂地语言表达。如有错误，欢迎指正
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        #这块定义了一个新的值，用作展示训练的效果
        #它的定义为：预测对的 / 总预测数，例如：0.55表示预测正确了55%
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #用以保存参数的函数（跑完下次再跑，就可以直接读取上次跑的结果而不必从头开始）
        saver = tf.train.Saver(tf.global_variables())
        
        #tf要求必须如此定义一个init变量，用以在初始化运行（也就是没有保存模型）时加载各个变量
        init = tf.global_variables_initializer()        
        #设置tf按需使用GPU资源，而不是有多少就占用多少
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config = config)
        #使用with，保证执行完后正常关闭tf
        with sess and open (pre_path, "w") as f:
            try:
                #定义了存储模型的文件路径，即：当前运行的python文件路径下，文件名为stock_rnn.ckpt
                saver.restore(sess, model_save_path)
                print ("成功加载模型参数")
            except:
                #如果是第一次运行，通过init告知tf加载并初始化变量
                print ("未加载模型参数，文件被删除或者第一次运行")
                sess.run(init)
            
            i = 0
            while True:
                train_x, train_y = self.get_train_softmax(file_path = train_path, time_step = origin_data_row, rtype="train")
                if train_x is None:
                    print ("训练集均已训练完毕")                    
                    saver.save(sess, model_path)
                    print("保存模型\n")
                    break

                if (i + 1) % 10 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        input_x:train_x, input_y: train_y, keep_prob: 1.0, batch_size: self.batch_size})
                    print ("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
                    saver.save(sess, model_path)
                    print("保存模型\n")
                    #这部分代码作用为：每次保存模型，顺便将预测收益和真实收益输出保存至show_y_softmax.txt文件下。熟悉tf可视化，完全可以采用可视化替代
                    _y_pre_train = sess.run(y_pre, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size:  self.batch_size})
                    _loss = sess.run(loss, feed_dict={
                        input_x:train_x, input_y: train_y, keep_prob: 1.0, batch_size: self.batch_size})
                    a1 = np.array(train_y).reshape(self.batch_size, output_num)
                    b1 = np.array(_y_pre_train).reshape(self.batch_size, output_num)
                    f.write(str(a1.tolist()))
                    f.write("\n")
                    f.write(str(b1.tolist()))
                    f.write("\n")
                    f.write(str(_loss))
                    f.write("\n")
                i += 1
                #按照给定的参数训练一次LSTM神经网络
                sess.run(train_op, feed_dict={input_x: train_x, input_y: train_y, keep_prob: 0.6, batch_size: self.batch_size})

            #计算测试数据的准确率
            #读取测试集数据
            test_x, test_y = self.get_train_softmax(file_path = train_path, time_step = origin_data_row, rtype="test")
            print ("test accuracy {0}".format(sess.run(accuracy, feed_dict={
                input_x: test_x, input_y: test_y, keep_prob: 1.0, batch_size:self.test_size})))
            self.init()