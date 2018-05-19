# encoding: UTF-8

import numpy as np
import re
import itertools
from collections import Counter
import os
import time, random
import pickle
import pandas as pd


import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

import multiprocessing
from gensim.models import Word2Vec

import psutil

class DataHelper:
    @classmethod
    def classify(cls, y):
        y = float(y)
        if y <= 0:
            return [1,0]
        else:
            return [0,1]

    @classmethod
    def get_text_train_data(cls, file=None, batch_size=10,  reduce_num=0, test_part_start=0.9, test_part_end=1):
        if file is None:
            return None, None, None
        if test_part_start < 0:
            test_part_start = 0.9
        if test_part_end < test_part_end:
            test_part_end = test_part_start + 0.1 if (test_part_start + 0.1 < 1) else 1

        data_temp = pd.read_csv(file)
        data_temp['0'] = data_temp['0'].apply(eval)
        data_temp['1'] = data_temp['1'].apply(eval)
        data_temp["0"] = data_temp["0"].apply(lambda x: x[0])
        data_temp["1"] = data_temp["1"].apply(lambda x: x[0])
        data_temp["1"] = data_temp["1"].apply(lambda x: cls.classify(x))

        x = data_temp["0"].tolist()[reduce_num:]
        y = data_temp["1"].tolist()[reduce_num:]
        y = np.array(y).reshape(len(y) , len(cls.classify(0))).tolist()

        max_word_num = 0
        for sentence in x:
            wn = sentence.count(' ') + 1
            if wn > max_word_num:
                max_word_num = wn

        data_length = len(x)
        train_length_1 = 0
        train_length_2 = int(data_length * test_part_start)
        train_length_3 = int(data_length * test_part_end)
        train_length_4 = data_length

        batch_index = []
        x_train = x[train_length_1:train_length_2] + x[train_length_3:train_length_4]
        y_train = y[train_length_1:train_length_2] + y[train_length_3:train_length_4]

        train_length = len(x_train)
        for i in range(train_length):
            if i % batch_size==0:
               batch_index.append(i)
        batch_index.append(train_length)

        index = []
        for i in range(len(x_train)):
            index.append(i)
        random.shuffle(index)
        x_random = []
        y_random = []
        for i in range(len(index)):
            x_random.append(x_train[i])
            y_random.append(y_train[i])

        return np.array(x_random), np.array(y_random), batch_index, max_word_num

    @classmethod
    def get_text_test_data(cls, file=None, reduce_num=0, test_part_start=0.9, test_part_end=1):
        if file is None:
            return None, None, None
        if test_part_start < 0:
            test_part_start = 0.9
        if test_part_end < test_part_end:
            test_part_end = test_part_start + 0.1 if (test_part_start + 0.1 < 1) else 1

        data_temp = pd.read_csv(file)
        data_temp['0'] = data_temp['0'].apply(eval)
        data_temp['1'] = data_temp['1'].apply(eval)
        data_temp["0"] = data_temp["0"].apply(lambda x: x[0])
        
        y = data_temp["1"].apply(lambda x: x[0])
        y = y.apply(lambda x: cls.classify(x))

        x = np.array(data_temp["0"].tolist()[reduce_num:])
        y = y.tolist()[reduce_num:]
        y = np.array(y).reshape(len(y) , len(cls.classify(0)))

        max_word_num = 0
        for sentence in x:
            wn = sentence.count(' ') + 1
            if wn > max_word_num:
                max_word_num = wn

        data_length = len(x)
        test_length_1 = int(data_length * test_part_start)
        test_length_2 = int(data_length * test_part_end)
        
        x = x[test_length_1:test_length_2]
        y = y[test_length_1:test_length_2]

        y_temp = data_temp["1"].tolist()[reduce_num:]
        y_others = []
        for d in y_temp[test_length_1:test_length_2]:
            y_others.append(d)
        return x, y, y_others, max_word_num

    @classmethod
    def get_number_data(cls, file=None, batch_size=10, reduce_num=0, test_part_start=0.9, test_part_end=1, stand=False):
        if file is None:
            return None, None, None

        data_temp = pd.read_csv(file)
        data_temp['0'] = data_temp['0'].apply(eval)
        data_temp['1'] = data_temp['1'].apply(eval)

        x_temp = data_temp["0"].tolist()[reduce_num:]
        clas = data_temp["1"].apply(lambda x: cls.classify(x[0])).tolist()[reduce_num:]
        y_temp = data_temp["1"].tolist()[reduce_num:]

        if stand:            
            x_t = []
            for x in x_temp:
                x = np.array(x, dtype = 'float_')
                mean = np.mean(x, axis=0)
                std = np.std(x, axis=0)
                x_t.append((x - mean) / std) #标准化
            x_temp = x_t

        data = []
        for i in range(len(x_temp)):
            data.append([x_temp[i], clas[i], y_temp[i]])        

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
            test_others.append(test_data[i][2])
        return np.array(x_train), np.array(y_train), np.array(batch_index), np.array(x_test), np.array(y_test), np.array(test_others)

    @classmethod
    def padding_sentences(cls, input_sentences, padding_token, padding_sentence_length = None):
        sentences = [sentence.split(' ') for sentence in input_sentences]
        max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
        for sk in range(0,len(sentences)):
            if len(sentences[sk]) > max_sentence_length:
                sentences[sk] = sentences[sk][:max_sentence_length]
            else:
                sentences[sk].extend([padding_token] * (max_sentence_length - len(sentences[sk])))
        return (sentences, max_sentence_length)

    @classmethod
    def embedding_sentences(cls, sentences, embedding_size = 128, window = 5, min_count = 5, file_to_load = None, file_to_save = None):
        if file_to_load is not None:
            w2vModel = Word2Vec.load(file_to_load)
        else:
            w2vModel = Word2Vec(sentences, size = embedding_size, window = window, min_count = min_count, workers = multiprocessing.cpu_count())
            if file_to_save is not None:
                w2vModel.save(file_to_save)
        all_vectors = []
        embeddingDim = w2vModel.vector_size
        embeddingUnknown = [0 for i in range(embeddingDim)]
        for sentence in sentences:
            this_vector = []
            for word in sentence:
                if word in w2vModel.wv.vocab:
                    this_vector.append(w2vModel[word])
                else:
                    this_vector.append(embeddingUnknown)
            all_vectors.append(this_vector)
        return all_vectors

    @classmethod    
    def make_word2vec_model(cls, basic_path, input_file, file_to_save = None, embedding_size = 128, step = 1, window = 5, min_count = 5):
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if input_file is None or file_to_save is None:
            return None

        input_path = os.path.join(basic_path, input_file)
        output_path = os.path.join(basic_path, file_to_save)

        data_temp = pd.read_csv(input_path, encoding='utf-8')        
        data = data_temp["1"].tolist()
        del data_temp

        def getMemCpu():
            data = psutil.virtual_memory()
            total = data.total #总内存,单位为byte
            free = data.available #可以内存
            memory =  "Memory usage:%d"%(int(round(data.percent)))+"%"+"  "
            cpu = "CPU:%0.2f"%psutil.cpu_percent(interval=1)+"%"
            return memory+cpu        

        sentences = []
        length = len(data)
        for i in range(length):
            if i % step != 0:
                continue

            sentences.append(str(data[0]).split(' '))
            del data[0]

            if i % 10000 == 0:
                print(getMemCpu())
                print(i)
        del data

        print('make word2vec model')
        DataHelper.embedding_sentences(sentences, embedding_size, window, min_count, file_to_save = output_path)

    @classmethod
    def rateCalc(cls, predictions, right_data, buy_when=2, start_date=None, end_date=None):
        sindex = 0
        eindex = len(predictions)
        if start_date != None:
            for i in range(len(predictions)):
                if right_data[i][1] >= start_date:
                    sindex = i
                    break
        if end_date != None:
            for ii in range(i, len(predictions)):
                if right_data[ii][1] >= end_date:
                    eindex = ii
                    break
        predictions = predictions[sindex:eindex]
        right_data = right_data[sindex:eindex]

        sum_rate = 0
        sum_all_rate = 0
        if (eindex > 0):
            for i in range(len(predictions)):
                if predictions[i] == buy_when:
                    sum_rate = sum_rate + float(right_data[i][0]) * 1.0
                sum_all_rate = sum_all_rate + float(right_data[i][0]) * 1.0        
            print( "%s 到 %s，总收益率为：%f，一直购买收益率为：%f，总交易天数为：%d" % (right_data[0][1], right_data[-1][1], sum_rate, sum_all_rate, len(right_data)) )
        return sum_rate, sum_all_rate