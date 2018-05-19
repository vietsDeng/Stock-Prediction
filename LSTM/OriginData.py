#coding=utf-8
import pymysql
import requests
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time, datetime
import random

from tensorflow.contrib import rnn
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Spider.tool import VTool

class OriginData():
    groupby_skip = False
    
    @classmethod
    def makeOriginDataCsv(cls, cur=None, start_date=None, end_date=None, basic_path=None, output_file=None, stock_id=None):
        #初始化源文件路径和存储文件路径        
        if cur is None or start_date is None or end_date is None or output_file is None or stock_id is None :
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))        
        output_path = os.path.join(basic_path, output_file)
        VTool.makeDirs(files=[output_path])

        data = cur.execute("select id, stock_id, date, opening, closing, difference, percentage_difference, lowest, highest, volume, amount from history where stock_id = '%s' and date between '%s' and '%s' " % (stock_id, start_date, end_date))
        data = cur.fetchall()
        if len(data) == 0:
            return None

        res = []
        for d in data:
            res.append([int(d[0]), int(d[1]), str(d[2]), float(d[3]), float(d[4]), float(d[5]), float(d[6]), float(d[7]), float(d[8]), float(d[9]), float(d[10])])
        new_data=[]
        for d in zip(*res):
            new_data.append(d)        
        origin_data = {'id':new_data[0], 'stock_id':new_data[1], 'date':new_data[2], 'opening':new_data[3], 'closing':new_data[4], 'difference':new_data[5], 'percentage_difference':new_data[6], 'lowest':new_data[7], 'highest':new_data[8], 'volume':new_data[9], 'amount':new_data[10]}

        #读取原始数据，只保留需要使用的列        
        total_data = DataFrame(origin_data)
        total_data.sort_values(by = ['stock_id', 'date'], inplace = True)
        #根据股票代码分组
        g_stock_num = total_data.groupby(by = ["stock_id"])
        total_data["rate"] = 100 * (g_stock_num.shift(0)["closing"] / g_stock_num.shift(1)["closing"] - 1)
        for i in total_data.index:
            total_data.loc[i, 'rate'] = str(np.round(float(total_data['rate'][i]), 2))
        #重新调整列的顺序，为接下来处理成输入、输出形式做准备
        columns = ["stock_id", "date", "opening", "closing", "difference", "percentage_difference", "lowest", "highest", "volume", "amount", "rate"]
        total_data = total_data[columns]
                            
        def func_train_data(data_one_stock_num):
            if cls.groupby_skip == False:
                cls.groupby_skip = True
                return None
            print ("正在处理的股票代码:%06s"%data_one_stock_num.name)
            data = {"stock_id":[], "date":[], "opening":[], "closing":[], "difference":[], "percentage_difference":[], "lowest":[], "highest":[], "volume":[], "amount":[], "rate":[]}
            for i in range(len(data_one_stock_num.index) - 1):
                for k in data:
                    data[k].append(data_one_stock_num.iloc[i][k])
            pd.DataFrame(data).to_csv(output_path, index=False, columns=columns)

        total_data1 = total_data.dropna()
        total_data2 = total_data1.drop(total_data1[(total_data1.rate == 'nan')].index)        
        g_stock_num = total_data2.groupby(by = ["stock_id"])
        #清空接收路径下的文件，初始化列名        
        cls.groupby_skip = False
        g_stock_num.apply(func_train_data)

    @classmethod
    def getImportVocab(cls, cur=None, count=50, ranking_type="tfidf"):
        if ranking_type not in ["tfidf", "textrank"]:
            ranking_type = "tfidf"

        if ranking_type == "tfidf":
            cur.execute("SELECT word FROM vocab where tfidf_ranking is not null order by tfidf_ranking limit %d " % (count))
        elif ranking_type == "textrank":
            cur.execute("SELECT word FROM vocab where textrank_ranking is not null order by textrank_ranking limit %d " % (count))
        data = cur.fetchall()

        words = []
        for d in data:
            words.append(d[0])
        return words      
    
    @classmethod
    def makeBindexDataCsv(cls, cur=None, start_date=None, end_date=None, basic_path=None, output_file=None, word_count=20, stock_id=None, ranking_type='tfidf'):
        if cur == None or start_date == None or end_date == None or output_file == None or stock_id == None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if word_count < 0:
            word_count = 20
        if ranking_type not in ["tfidf", "textrank"]:
            ranking_type = "tfidf"
        output_path = os.path.join(basic_path, output_file)
        VTool.makeDirs(files=[output_path])

        words = cls.getImportVocab(cur, count=20, ranking_type=ranking_type)
        word_count = len(words)
        for i in range(len(words)):
            words[i] = "'" + words[i] + "'"
        words_str = ",".join(words)
        del words

        word_key_list = []
        for i in range(1,word_count+1):
            word_key_list.append("word%s" % i)
        columns = ["stock_id", "date", "opening", "closing", "difference", "percentage_difference", "lowest", "highest", "volume", "amount", "rate"] + word_key_list
        data = {}
        for k in columns:
            data[k] = []
        pd.DataFrame(data).to_csv(output_path, index=False, columns=columns)

        cur.execute("SELECT count(*) as count FROM history WHERE stock_id = '%s' and date between '%s' and '%s' " % (stock_id, start_date, end_date))
        count = cur.fetchall()
        count = count[0][0]     
        
        skip = 50
        slimit = 0
        while slimit < count:
            cur.execute("SELECT stock_id, opening, closing, difference, percentage_difference, lowest, highest, volume, amount, date FROM history WHERE stock_id = '%s' and date between '%s' and '%s' order by date asc, stock_id asc limit %d,%d " % (stock_id, start_date, end_date, 0 if slimit-1 < 0 else slimit-1, skip if slimit-1 < 0 else skip+1))
            slimit += skip
            history_tt = cur.fetchall()
            history_t = []
            for h in history_tt:
                history_t.append([int(h[0]), float(h[1]), float(h[2]), float(h[3]), float(h[4]), float(h[5]), float(h[6]), float(h[7]), float(h[8]), str(h[9])])
            del history_tt
            
            sdate = str(history_t[0][9])
            edate = str(history_t[-1][9])
            sdate = datetime.datetime.strptime(sdate,'%Y-%m-%d')
            sdate = (sdate - datetime.timedelta( days=1 )).strftime('%Y-%m-%d')            
            cur.execute("SELECT b.vocab_id, b.bindex, b.date FROM vocab v left join baidu_index b on v.id = b.vocab_id WHERE v.word in (%s) and b.date between '%s' and '%s' order by date, vocab_id asc" % (words_str, sdate, edate))
            bindex = cur.fetchall()
            bindex_t = []            
            bindex_vec = 0
            cur_date = None
            if len(bindex) > 0:
                cur_date = str(bindex[0][2])
            bix = []
            bix_item = [cur_date]
            if len(bindex) > 0:
                for bi in bindex:
                    if str(bi[2]) != cur_date:                    
                        cur_date = str(bi[2])
                        bix.append(bix_item)
                        bix_item = [cur_date]
                    bix_temp = json.loads(bi[1])
                    bix_item.append(bix_temp['all']['0'])
                bix.append(bix_item)
            del bindex

            bindex = {}
            for k in range(1,len(bix)):
                b_t = []
                for kk in range(1,len(bix[k])):
                    if int(bix[k][kk]) != 0 and int(bix[k-1][kk]) != 0:
                        b_t.append(str(np.round(float(100 * (int(bix[k][kk]) / int(bix[k-1][kk]) - 1)), 2)))
                    else:
                        b_t.append(str(0.01))
                bindex[bix[k][0]] = b_t
            del bix

            for i in range(len(history_t)):
                history_t[i] += bindex[history_t[i][9]]
            history_temp = []
            for h in zip(*history_t):
                history_temp.append(h)
            history = {'stock_id':history_temp[0], 'opening':history_temp[1], 'closing':history_temp[2], 'difference':history_temp[3], 'percentage_difference':history_temp[4], 'lowest':history_temp[5], 'highest':history_temp[6], 'volume':history_temp[7], 'amount':history_temp[8], 'date':history_temp[9]}
            for i in range(10, 10+word_count):
                history["word%s" % (i-9)] = history_temp[i]
            del history_t, history_temp        
            history = DataFrame(history)
            g_history = history.groupby(by = ['stock_id'])
            #0.01 -> 1 % 保留2位小数
            history['rate'] = 100 * (g_history.shift(0)["closing"] / g_history.shift(1)["closing"] - 1)
            history.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
            for i in history.index:
                history.loc[i, 'rate'] = str(np.round(float(history['rate'][i]), 2))            

            #将经过标准化的数据处理成训练集和测试集可接受的形式              
            def func_train_data(data_stock):
                if cls.groupby_skip == False:
                    cls.groupby_skip = True
                    return None
                print ("正在处理的股票代码:%06s"%data_stock.name)
                
                data = {}
                for k in columns:
                    data[k] = []
                for i in range(len(data_stock) - 1):
                    for k in data:
                        data[k].append(data_stock.iloc[i][k])
                pd.DataFrame(data).to_csv(output_path, index=False, header=False, mode="a", columns=columns)
            
            g_stock = history.groupby(by = ["stock_id"])
            #清空接收路径下的文件，初始化列名                       
            cls.groupby_skip = False
            g_stock.apply(func_train_data)

    @classmethod
    def makeNewsDataCsv(cls, cur=None, start_date=None, end_date=None, basic_path=None, word_trend_file=None, news_file=None, output_file=None, stock_id=None):
        if cur == None or start_date == None or end_date == None or word_trend_file is None or output_file == None or stock_id == None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        news_path = os.path.join(basic_path, news_file)
        word_trend_path = os.path.join(basic_path, word_trend_file)
        output_path = os.path.join(basic_path, output_file)
        VTool.makeDirs(files=[output_path])

        columns = ["stock_id", "date", "opening", "closing", "difference", "percentage_difference", "lowest", "highest", "volume", "amount", "rate"] + ["news_pos_num", "news_neg_num"]
        data = {}
        for k in columns:
            data[k] = []
        pd.DataFrame(data).to_csv(output_path, index=False, columns=columns)        

        word_trend = {}
        word_trend_temp = pd.read_csv(word_trend_path)
        for k in word_trend_temp["0"].keys():
            word_trend[word_trend_temp["0"][k]] = [word_trend_temp["1"][k], word_trend_temp["2"][k]]
        p_up = word_trend['total_words'][0] / (word_trend['total_words'][0] + word_trend['total_words'][1])
        p_down = word_trend['total_words'][1] / (word_trend['total_words'][0] + word_trend['total_words'][1])

        cur.execute("SELECT count(*) as count FROM history WHERE stock_id = '%s' and date between '%s' and '%s' " % (stock_id, start_date, end_date))
        count = cur.fetchall()
        count = count[0][0]     

        skip = 100
        slimit = 0
        while slimit < count:
            cur.execute("SELECT stock_id, opening, closing, difference, percentage_difference, lowest, highest, volume, amount, date FROM history WHERE stock_id = '%s' and date between '%s' and '%s' order by date asc, stock_id asc limit %d,%d " % (stock_id, start_date, end_date, 0 if slimit-1 < 0 else slimit-1, skip if slimit-1 < 0 else skip+1))
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
            # sdate = datetime.datetime.strptime(sdate,'%Y-%m-%d')
            # sdate = (sdate - datetime.timedelta(days=0)).strftime('%Y-%m-%d')
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
            def func_train_data(data_stock):
                if cls.groupby_skip == False:
                    cls.groupby_skip = True
                    return None
                print ("正在处理的股票代码:%06s"%data_stock.name)

                data = {}
                for k in columns:
                    data[k] = []
                for i in range(len(data_stock) - 1):
                    for k in data:
                        data[k].append(data_stock.iloc[i][k])
                pd.DataFrame(data).to_csv(output_path, index=False, header=False, mode="a", columns=columns)                
            
            g_stock = history.groupby(by = ["stock_id"])
            #清空接收路径下的文件，初始化列名                       
            cls.groupby_skip = False
            g_stock.apply(func_train_data)