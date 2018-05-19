#coding=utf-8

'''
CNN 文本分类之中文预处理(分词)
分词提取实词后，词汇量仍然相对较大，在样本和计算资源有限的情况下，可以考虑选取频率较高的特征作为原始特征
'''

import jieba
import jieba.posseg as pseg
from jieba import  analyse
import os
import json
import datetime
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import subprocess

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Spider.tool import VTool

"""
制作股票分类数据
"""
class OriginData:
    flag_list = ['t','q','p','u','e','y','o','w','m']
    groupby_skip = False

    @classmethod
    def jiebafenci(cls, all_the_text):    
        re = ""
        words = pseg.cut(all_the_text)
        for w in words:
            flag = w.flag
            tmp = w.word
            if len(tmp)>1 and len(flag)>0 and flag[0] not in cls.flag_list and  tmp[0]>=u'/u4e00' and tmp[0]<=u'\u9fa5':
                re = re + " " + w.word
        re = re.replace("\n"," ").replace("\r"," ").strip()

        return re

    """
    制作"股票-新闻标题"分类数据
    """
    @classmethod
    def makeTitleOriginCsv(cls, cur=None, start_date=None, end_date=None, day_num=1, basic_path=None, input_file=None, output_file=None, stock_id=None):
        #初始化源文件路径和存储文件路径
        if cur is None or start_date is None or end_date is None or stock_id is None or input_file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if output_file is None:
            output_file = "text_data.csv"
        input_path = os.path.join(basic_path, input_file)
        output_path = os.path.join(basic_path, output_file)
        VTool.makeDirs(files=[output_path])

        #清空接收路径下的文件，初始化列名
        pd.DataFrame({"0":[], "1":[]}).to_csv(output_path, index=False, encoding="utf-8")

        cur.execute("SELECT count(*) as count FROM history WHERE stock_id = '%s' and date between '%s' and '%s' " % (stock_id, start_date, end_date))
        count = cur.fetchall()
        count = count[0][0]

        deviation = 2        
        skip = 50
        slimit = 0
        while slimit < count:
            cur.execute("SELECT stock_id,closing,date FROM history WHERE stock_id = '%s' and date between '%s' and '%s' order by date, stock_id asc limit %d,%d " % (stock_id, start_date, end_date, 0 if slimit-deviation-day_num < 0 else slimit-deviation-day_num, skip if slimit-deviation-day_num < 0 else skip+deviation+day_num))
            history_t = cur.fetchall()
            sdate = str(history_t[0][2])
            edate = str(history_t[-1][2])

            history_temp = []
            for h in zip(*history_t):
                history_temp.append(h)
            history = {'stock_id':history_temp[0], 'closing':history_temp[1], 'date':history_temp[2]}
            del history_t, history_temp        
            history = DataFrame(history)
            g_history = history.groupby(by = ['stock_id'])
            #0.01 -> 1 % 保留2位小数
            history['rate'] = 100 * (g_history.shift(0)["closing"] / g_history.shift(1)["closing"] - 1)
            history.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
            '''
            '''
            cur.execute("SELECT GROUP_CONCAT(id SEPARATOR ','), time FROM news WHERE time between '%s' and '%s' GROUP BY time order by time " % (sdate, edate))
            news_temp = cur.fetchall()            

            news_by_date = {}
            news_by_id = {}
            for n in news_temp:
                news_by_date[str(n[1])] = n[0].split(",")
                for nid in news_by_date[str(n[1])]:
                    news_by_id[nid] = ''
            del news_temp

            nid_len = len(news_by_id)
            reader = pd.read_csv(input_path, chunksize=1000)
            for sentences in reader:
                for k in sentences['1'].keys():
                    nid = str(sentences['0'][k])
                    if nid in news_by_id and news_by_id[nid] == '':
                        news_by_id[nid] = str(sentences['1'][k]).split(" ")
                        nid_len-=1
                if nid_len <= 0:
                    break
            reader.close()
            del reader, sentences

            news_date = {}
            for k in history['date'].keys():
                if (k-deviation-day_num+1) in history['date']:
                    news_date[str(history['date'][k])] = [str(history['date'][k-deviation-day_num+1]), str(history['date'][k-deviation])]

            def func_train_data(date_stock):                
                if cls.groupby_skip == False:
                    cls.groupby_skip = True
                    return None

                date = str(date_stock.name)
                if date not in news_date:
                    return                
                sdate = datetime.datetime.strptime(news_date[date][0], '%Y-%m-%d')
                edate = datetime.datetime.strptime(news_date[date][1], '%Y-%m-%d')

                words = []
                while sdate <= edate:
                    cur_date = sdate.strftime('%Y-%m-%d')
                    sdate += datetime.timedelta(days=1)
                    if cur_date not in news_by_date:
                        print("%s error" % cur_date)
                        return None
                    for i in news_by_date[cur_date]:
                        words += news_by_id[i]
                
                data = []
                for k in date_stock['stock_id'].keys():
                    data.append([[" ".join(words)], [str(np.round(float(history['rate'][k]), 2)), str(date_stock['date'][k]), str(date_stock['stock_id'][k])]])                

                print ("正在处理的日期:%s"%date_stock.name)            
                pd.DataFrame(data).to_csv(output_path, index=False, header=False, mode="a", encoding="utf-8")

            g_stock = history.groupby(by = ["date"])
            cls.groupby_skip = False
            g_stock.apply(func_train_data)
            slimit += skip                

    """
    新闻处理
    """
    @classmethod
    def makeNewsKeywordCacheCsv(cls, cur=None, start_date=None, end_date=None, basic_path=None, analyse_type='tfidf', rewrite=True):
        #初始化源文件路径和存储文件路径
        if cur is None or start_date is None or end_date is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if analyse_type not in ['tfidf', 'textrank', 'all', 'title']:
            return None    
        tfidf = analyse.extract_tags
        textrank = analyse.textrank

        origin_data_path = os.path.join(basic_path, "%s_keyword_cache.csv" % analyse_type)
        VTool.makeDirs(files=[origin_data_path])
        #清空接收路径下的文件，初始化列名
        if rewrite == True:
            pd.DataFrame({"0":[], "1":[]}).to_csv(origin_data_path, index=False, encoding="utf-8")
            
        skip = 30
        start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
        start_date -= datetime.timedelta(days=1)
        end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')
        i = 1
        while start_date <= end_date:
            start_date += datetime.timedelta(days=1)
            cur_date = start_date.strftime('%Y-%m-%d')
            start_date += datetime.timedelta(days=skip)
            if start_date > end_date:
                cur_end_date = end_date.strftime('%Y-%m-%d')
            else:
                cur_end_date = start_date.strftime('%Y-%m-%d')
            
            if analyse_type == 'title':
                cur.execute("SELECT id, title FROM news WHERE time between '%s' and '%s' order by time, title" % (cur_date, cur_end_date))
            else:
                cur.execute("SELECT id, content FROM news WHERE time between '%s' and '%s' order by time, content" % (cur_date, cur_end_date))
            news = cur.fetchall()
            news_keyword = []
            for n in news:
                i+=1
                print(i)
                if analyse_type == 'tfidf':
                    kword = tfidf(n[1], allowPOS=['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v'])
                    kword = " ".join(kword)
                elif analyse_type == 'textrank':
                    kword = textrank(n[1], allowPOS=['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v'])
                    kword = " ".join(kword)
                elif analyse_type == 'all':
                    kword = cls.jiebafenci(n[1])
                elif analyse_type == 'title':
                    kword = cls.jiebafenci(n[1])
                else:
                    kword = ''
                keywords = [str(n[0]), kword.strip()]
                news_keyword.append(keywords)
            pd.DataFrame(news_keyword).to_csv(origin_data_path, index=False, header=False, mode="a", encoding="utf-8")

    @classmethod
    def makeImportVocab(cls, basic_path=None, keyword_csv_file=None, important_vocab_csv_file=None):
        #初始化源文件路径和存储文件路径    
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if keyword_csv_file is None:
            return None
        if important_vocab_csv_file is None:
            return None
        input_data_path = os.path.join(basic_path, keyword_csv_file)
        output_data_path = os.path.join(basic_path, important_vocab_csv_file)
        VTool.makeDirs(files=[output_data_path])
        #清空接收路径下的文件，初始化列名
        pd.DataFrame({"0":[], "1":[]}).to_csv(output_data_path, index=False, encoding="utf-8")
            
        i = 0
        vocab = {}
        reader = pd.read_csv(input_data_path, chunksize=5000)
        for sentences in reader:
            for sentence in sentences['1']:
                i += 1
                print(i)
                if str(sentence) == 'nan':
                    continue
                words = sentence.split(" ")            
                for word in words:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1
        sorted_vocab = sorted(vocab.items(), key=lambda v : v[1], reverse=True)    

        data = []
        for word, num in sorted_vocab:
            data.append([word, num])
        if len(data) != 0:
            pd.DataFrame(data).to_csv(output_data_path, index=False, header=False, mode="a", encoding="utf-8")

    @classmethod
    def importWordToMysql(cls, cur=None, basic_path=None, tfidf_file=None, textrank_file=None, word_count=100):
        #初始化源文件路径和存储文件路径
        if cur is None or tfidf_file is None or textrank_file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))

        tfidf_file_path = os.path.join(basic_path, tfidf_file)
        textrank_file_path = os.path.join(basic_path, textrank_file)

        cursor = pd.read_csv(tfidf_file_path, iterator = True)
        tfidf_data = cursor.get_chunk(word_count)
        cursor = pd.read_csv(textrank_file_path, iterator = True)
        textrank_data = cursor.get_chunk(word_count)

        vocabs ={}
        for k in tfidf_data['0'].keys():
            if tfidf_data['0'][k] not in vocabs:
                vocabs[tfidf_data['0'][k]] = [tfidf_data['0'][k], None, None]
            vocabs[tfidf_data['0'][k]][1] = k+1

        for k in textrank_data['0'].keys():
            if textrank_data['0'][k] not in vocabs:
                vocabs[textrank_data['0'][k]] = [textrank_data['0'][k], None, None]
            vocabs[textrank_data['0'][k]][2] = k+1

        sql = "insert into vocab(word, tfidf_ranking, textrank_ranking) values (%s, %s, %s)"
        for k in vocabs:
            cur.execute(sql, vocabs[k])

    @classmethod
    def importCityToMysql(cls, cur=None):        
        if cur is None:
            return None

        citys = [['全国', 0], ['北京', 514], ['上海', 57], ['广州', 95], ['深圳', 94]]
        sql = "insert into vocab(word, baidu_code) values (%s, %s)"
        for city in citys:
            cur.execute(sql, city)

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

    """
    制作"股票-百度指数"分类数据
    """
    @classmethod
    def makeBindexOriginCsv(cls, cur=None, words=None, start_date=None, end_date=None, day_num=1, basic_path=None, output_file=None, stock_id=None):
        #初始化源文件路径和存储文件路径
        if cur is None or words is None or start_date is None or end_date is None or stock_id is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if output_file is None:
            output_file = "bindex_data.csv"
        output_path = os.path.join(basic_path, output_file)
        VTool.makeDirs(files=[output_path])
        
        #清空接收路径下的文件，初始化列名
        pd.DataFrame({"0":[], "1":[]}).to_csv(output_path, index=False, encoding="utf-8")
            
        start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')

        for i in range(len(words)):
            words[i] = "'" + words[i] + "'"
        words_str = ",".join(words)

        cur.execute("SELECT count(*) as count FROM history WHERE stock_id = '%s' and date between '%s' and '%s' " % (stock_id, start_date, end_date))
        count = cur.fetchall()
        count = count[0][0]

        deviation = 2
        skip = 100
        slimit = 0
        while slimit < count:
            cur.execute("SELECT stock_id,closing,date FROM history WHERE stock_id = '%s' and date between '%s' and '%s' order by date, stock_id asc limit %d,%d " % (stock_id, start_date, end_date, 0 if slimit-deviation-day_num < 0 else slimit-deviation-day_num, skip if slimit-deviation-day_num < 0 else skip+deviation+day_num))
            history_t = cur.fetchall()
            sdate = str(history_t[0][2])
            edate = str(history_t[-1][2])

            history_temp = []
            for h in zip(*history_t):
                history_temp.append(h)
            history = {'stock_id':history_temp[0], 'closing':history_temp[1], 'date':history_temp[2]}
            del history_t, history_temp        
            history = DataFrame(history)
            g_history = history.groupby(by = ['stock_id'])
            #0.01 -> 1 % 保留2位小数
            history['rate'] = 100 * (g_history.shift(0)["closing"] / g_history.shift(1)["closing"] - 1)
            history.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)            
            '''
            '''
            cur.execute("SELECT b.vocab_id, b.bindex, b.date FROM vocab v left join baidu_index b on v.id = b.vocab_id WHERE v.word in (%s) and b.date between '%s' and '%s' order by date, vocab_id asc" % (words_str, sdate, edate))
            bindex = cur.fetchall()
            news_date = {}
            for k in history['date'].keys():
                if (k-deviation-day_num+1) in history['date']:
                    news_date[str(history['date'][k])] = [str(history['date'][k-deviation-day_num+1]), str(history['date'][k-deviation])]
            
            bindex_t = []
            bindex_vec = 0
            cur_date = None
            if len(bindex) > 0:
                cur_date = str(bindex[0][2])
            bix = []
            for bi in bindex:
                if str(bi[2]) != cur_date:                
                    bindex_t.append([bix, cur_date])
                    cur_date = str(bi[2])
                    bix = []
                
                bix_temp = json.loads(bi[1])            
                bix_temp = sorted(bix_temp.items(), key=lambda v : v[0])                    
                for k,b in bix_temp:
                    bix_list = sorted(b.items(), key=lambda v : v[0])
                    for kk,bb in bix_list:                
                        bix.append(bb)
                if bindex_vec == 0:
                    bindex_vec = len(bix)
            bindex_t.append([bix, cur_date])
            del bindex

            bindex_by_date = {}
            for k in range(1, len(bindex_t)):
                b_t = []
                for kk in range(len(bindex_t[k][0])):
                    if int(bindex_t[k][0][kk]) != 0 and int(bindex_t[k-1][0][kk]) != 0:
                        b_t.append(str(np.round(float(100 * (int(bindex_t[k][0][kk]) / int(bindex_t[k-1][0][kk]) - 1)), 2)))
                    else:
                        b_t.append(str(0.00))
                bindex_by_date[bindex_t[k][1]] = b_t
            del bindex_t

            def func_train_data(date_stock):
                if cls.groupby_skip == False:
                    cls.groupby_skip = True
                    return None

                date = str(date_stock.name)
                if date not in news_date:
                    return                
                sdate = datetime.datetime.strptime(news_date[date][0], '%Y-%m-%d')
                edate = datetime.datetime.strptime(news_date[date][1], '%Y-%m-%d')

                bindexs = []
                while sdate <= edate:
                    cur_date = sdate.strftime('%Y-%m-%d')
                    sdate += datetime.timedelta(days=1)
                    if cur_date not in bindex_by_date:
                        print("%s error" % cur_date)
                        exit()
                    else:
                        bindexs += bindex_by_date[cur_date]
                
                data = []
                for k in date_stock['stock_id'].keys():
                    data.append([(np.array(bindexs).reshape(int(len(bindexs)/bindex_vec), bindex_vec)).tolist(), [str(np.round(float(history['rate'][k]), 2)), str(date_stock['date'][k]), str(date_stock['stock_id'][k])]])                
                print ("正在处理的日期:%s"%date_stock.name)            
                pd.DataFrame(data).to_csv(output_path, index=False, header=False, mode="a", encoding="utf-8")

            g_stock = history.groupby(by = ["date"])
            cls.groupby_skip = False
            g_stock.apply(func_train_data)
            slimit += skip

    '''
    使用jiebe_TF-IDF/TEXTRANK获取新闻内容关键词，作训练输入
    '''
    @classmethod
    def makeTextOriginCsv(cls, cur=None, start_date=None, end_date=None, day_num=1, basic_path=None, input_file=None, output_file=None, stock_id=None, rewrite=True):
        #初始化源文件路径和存储文件路径
        if cur is None or start_date is None or end_date is None or input_file is None or output_file is None or stock_id is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))    
        input_path = os.path.join(basic_path, input_file)
        output_path = os.path.join(basic_path, output_file)
        VTool.makeDirs(files=[output_path])
        '''
        '''
        cur.execute("SELECT count(*) as count FROM history WHERE stock_id = '%s' and date between '%s' and '%s' " % (stock_id, start_date, end_date))
        count = cur.fetchall()
        count = count[0][0]
        if rewrite == True:
            pd.DataFrame({"0":[], "1":[]}).to_csv(output_path, index=False)
        
        deviation = 2
        skip = 50
        slimit = 0
        while slimit < count:
            cur.execute("SELECT stock_id,closing,date FROM history WHERE stock_id = '%s' and date between '%s' and '%s' order by date asc, stock_id asc limit %d,%d " % (stock_id, start_date, end_date, 0 if slimit-deviation-day_num < 0 else slimit-deviation-day_num, skip if slimit-deviation-day_num < 0 else skip+deviation+day_num))
            history_t = cur.fetchall()

            sdate = str(history_t[0][2])
            edate = str(history_t[-1][2])

            history_temp = []
            for h in zip(*history_t):
                history_temp.append(h)
            history = {'stock_id':history_temp[0], 'closing':history_temp[1], 'date':history_temp[2]}
            del history_t, history_temp        
            history = DataFrame(history)
            g_history = history.groupby(by = ['stock_id'])
            #0.01 -> 1 % 保留2位小数
            history['rate'] = 100 * (g_history.shift(0)["closing"] / g_history.shift(1)["closing"] - 1)
            history.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)            
            '''
            '''
            cur.execute("SELECT GROUP_CONCAT(id SEPARATOR ','), time FROM news WHERE time between '%s' and '%s' GROUP BY time order by time " % (sdate, edate))
            news_temp = cur.fetchall()

            news_date = {}
            for k in history['date'].keys():
                if (k-deviation-day_num+1) in history['date']:
                    news_date[str(history['date'][k])] = [str(history['date'][k-deviation-day_num+1]), str(history['date'][k-deviation])]

            news_by_date = {}
            news_by_id = {}
            for n in news_temp:
                news_by_date[str(n[1])] = n[0].split(",")
                for nid in news_by_date[str(n[1])]:
                    news_by_id[nid] = ''
            del news_temp

            nid_len = len(news_by_id)
            reader = pd.read_csv(input_path, chunksize=1000)
            for sentences in reader:
                for k in sentences['1'].keys():
                    nid = str(sentences['0'][k])
                    if nid in news_by_id and news_by_id[nid] == '':
                        news_by_id[nid] = str(sentences['1'][k]).split(" ")
                        nid_len-=1
                if nid_len <= 0:
                    break
            reader.close()
            del reader, sentences
            
            def func_train_data(date_stock):                
                if cls.groupby_skip == False:
                    cls.groupby_skip = True
                    return None

                date = str(date_stock.name)
                if date not in news_date:
                    return                
                sdate = datetime.datetime.strptime(news_date[date][0], '%Y-%m-%d')
                edate = datetime.datetime.strptime(news_date[date][1], '%Y-%m-%d')

                words = []
                while sdate <= edate:
                    cur_date = sdate.strftime('%Y-%m-%d')
                    sdate += datetime.timedelta(days=1)
                    if cur_date not in news_by_date:
                        print("%s error" % cur_date)
                        return None
                    for i in news_by_date[cur_date]:
                        words += news_by_id[i]
                
                data = []
                for k in date_stock['stock_id'].keys():
                    data.append([[" ".join(words)], [str(np.round(float(history['rate'][k]), 2)), str(date_stock['date'][k]), str(date_stock['stock_id'][k])]])                

                print ("正在处理的日期:%s"%date_stock.name)            
                pd.DataFrame(data).to_csv(output_path, index=False, header=False, mode="a", encoding="utf-8")

            g_stock = history.groupby(by = ["date"])                    
            cls.groupby_skip = False
            g_stock.apply(func_train_data)
            slimit += skip

    """
    制作"股票-历史数据"分类数据
    """    
    @classmethod
    def makeTrendStockOriginCsv(cls, cur=None, start_date=None, end_date=None, day_num=3, basic_path=None, stock_id=None, word_trend_file=None, news_file=None, output_file=None):
        #初始化源文件路径和存储文件路径
        if cur is None or start_date is None or end_date is None or stock_id is None or output_file is None or word_trend_file is None or news_file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))

        word_trend_path = os.path.join(basic_path, word_trend_file)
        news_path = os.path.join(basic_path, news_file)
        output_path = os.path.join(basic_path, output_file)        
        VTool.makeDirs(files=[output_path])
        #清空接收路径下的文件，初始化列名
        pd.DataFrame({"0":[], "1":[]}).to_csv(output_path, index=False, encoding="utf-8")        

        word_trend = {}
        word_trend_temp = pd.read_csv(word_trend_path)
        for k in word_trend_temp["0"].keys():
            word_trend[word_trend_temp["0"][k]] = [word_trend_temp["1"][k], word_trend_temp["2"][k]]
        p_up = word_trend['total_words'][0] / (word_trend['total_words'][0] + word_trend['total_words'][1])
        p_down = word_trend['total_words'][1] / (word_trend['total_words'][0] + word_trend['total_words'][1])

        cur.execute("SELECT count(*) as count FROM history WHERE stock_id = '%s' and date between '%s' and '%s' " % (stock_id, start_date, end_date))
        count = cur.fetchall()
        count = count[0][0]
        deviation = 2
        skip = 100
        slimit = 0        
        while slimit < count:
            cur.execute("SELECT stock_id, opening, closing, difference, percentage_difference, lowest, highest, volume, amount, date FROM history WHERE stock_id = '%s' and date between '%s' and '%s' order by date asc, stock_id asc limit %d,%d " % (stock_id, start_date, end_date, 0 if slimit-day_num-deviation < 0 else slimit-day_num-deviation, skip if slimit-day_num-deviation < 0 else skip+day_num+deviation))            
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
                
                data_temp_x = data_stock[["opening", "closing", "difference", "percentage_difference", "lowest", "highest", "volume", "amount", "news_pos_num", "news_neg_num"]]
                data_temp_y = data_stock[["rate", "date", "stock_id"]]
                data_res = []            
                for i in range(day_num - 1, len(data_temp_x.index) - deviation):
                    data_res.append( [data_temp_x.iloc[i - day_num + 1: i + 1].values.reshape(day_num, 10).tolist()] + data_temp_y.iloc[i + deviation].values.reshape(1,3).tolist() )                
                if len(data_res) != 0:
                    pd.DataFrame(data_res).to_csv(output_path, index=False, header=False, mode="a")                    

            g_stock_num = history.groupby(by = ["stock_id"])            
            cls.groupby_skip = False
            g_stock_num.apply(func_train_data)
            slimit += skip

    """
    制作"新闻[全文/tfidf/textrank]词数值化"分类数据
    """
    """
    @classmethod
    def makeWordNumOriginCsv(cls, cur=None, start_date=None, end_date=None, stock_id_str=None, basic_path=None, word_file=None, output_file=None, length=0, weight=0, day_num=1, rewrite=True):
        #初始化源文件路径和存储文件路径
        if cur is None or start_date is None or end_date is None or stock_id_str is None or word_file is None or output_file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))    
        word_path = os.path.join(basic_path, word_file)
        output_path = os.path.join(basic_path, output_file)
        VTool.makeDirs(files=[output_path])

        if length <= 0:
            cur.execute("SELECT max(num) FROM (SELECT count(id) as num FROM news WHERE time between '%s' and '%s' group by time) t " % (start_date, end_date))
            data = cur.fetchall()
            length = data[0][0]
        if weight <= 0:
            reader = pd.read_csv(word_path, chunksize=2000)
            for sentences in reader:
                for k in sentences['0'].keys():
                    w = len(str(sentences['1'][k]).split(" "))
                    if weight < w:
                        weight = w
            reader.close()

        word_trend = {}
        data = pd.read_csv("./data/word_trend.csv")
        mul = 1
        for k in data["0"].keys():
            if data["0"][k] != "total_words":
                word_trend[data["0"][k]] = [data["1"][k]*mul, data["2"][k]*mul*(-1)]
            else:
                word_trend[data["0"][k]] = [(1/data["1"][k])*mul, (1/data["2"][k])*mul*(-1)]
        del data

        '''
        '''
        cur.execute("SELECT count(*) as count FROM history WHERE stock_id in (%s) and date between '%s' and '%s' " % (stock_id_str, start_date, end_date))
        count = cur.fetchall()
        count = count[0][0]

        if rewrite == True:
            pd.DataFrame({"0":[], "1":[]}).to_csv(output_path, index=False)

        stock_id_num = len(stock_id_str.split(","))
        skip = 50 * stock_id_num
        slimit = 0
        while slimit < count:
            cur.execute("SELECT stock_id,closing,date FROM history WHERE stock_id in (%s) and date between '%s' and '%s' order by date asc, stock_id asc limit %d,%d " % (stock_id_str, start_date, end_date, 0 if slimit-stock_id_num < 0 else slimit-stock_id_num, skip if slimit-stock_id_num < 0 else skip+stock_id_num))
            history_t = cur.fetchall()
            history_temp = []
            for h in zip(*history_t):
                history_temp.append(h)
            history = {'stock_id':history_temp[0], 'closing':history_temp[1], 'date':history_temp[2]}
            del history_t, history_temp        
            history = DataFrame(history)
            g_history = history.groupby(by = ['stock_id'])
            #0.01 -> 1 % 保留2位小数
            history['rate'] = 100 * (g_history.shift(0)["closing"] / g_history.shift(1)["closing"] - 1)
            history.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

            sdate = str(history['date'][history['date'].keys()[0]])
            edate = str(history['date'][history['date'].keys()[-1]])
            sdate = datetime.datetime.strptime(sdate,'%Y-%m-%d')
            sdate = (sdate - datetime.timedelta(days=day_num)).strftime('%Y-%m-%d')
            '''
            '''
            cur.execute("SELECT GROUP_CONCAT(id SEPARATOR ','), time FROM news WHERE time between '%s' and '%s' GROUP BY time order by time " % (sdate, edate))
            news_temp = cur.fetchall()        
            news_by_date = {}
            news_by_id = {}
            for n in news_temp:
                news_by_date[str(n[1])] = n[0].split(",")
                for nid in news_by_date[str(n[1])]:
                    news_by_id[nid] = ''
            del news_temp        

            nid_len = len(news_by_id)
            reader = pd.read_csv(word_path, chunksize=1000)
            for sentences in reader:
                for k in sentences['1'].keys():
                    nid = str(sentences['0'][k])
                    if nid in news_by_id and news_by_id[nid] == '':
                        news_by_id[nid] = str(sentences['1'][k]).split(" ")                    
                        wt = []                    
                        for w in news_by_id[nid]:
                            if w in word_trend:
                                if word_trend[w][0] > 0:
                                    wt.append(word_trend[w][0])
                                else:
                                    wt.append(word_trend["total_words"][0])
                                if word_trend[w][1] > 0:
                                    wt.append(word_trend[w][1])
                                else:
                                    wt.append(word_trend["total_words"][1])
                            else:
                                wt.append(word_trend["total_words"][0])
                                wt.append(word_trend["total_words"][1])                    
                        if weight * 2 < len(wt):
                            wt = wt[0:weight*2]
                        elif weight * 2 > len(wt):
                            for i in range(int(len(wt)/2),weight):
                                wt.append(word_trend["total_words"][0])
                                wt.append(word_trend["total_words"][1])                    
                        news_by_id[nid] = wt
                        nid_len-=1
                if nid_len <= 0:
                    break
            reader.close()
            del reader, sentences        

            word_num_append = []
            for i in range(weight):
                word_num_append.append(word_trend["total_words"][0])
                word_num_append.append(word_trend["total_words"][1])         
            
            def func_train_data(date_stock):                
                if cls.groupby_skip == False:
                    cls.groupby_skip = True
                    return None            

                date = str(date_stock.name)
                date = datetime.datetime.strptime(date,'%Y-%m-%d')
                edate = date - datetime.timedelta(days=1)
                sdate = date - datetime.timedelta(days=day_num)

                word_num = []
                d = 0
                while sdate <= edate:
                    d += 1
                    cur_date = sdate.strftime('%Y-%m-%d')
                    sdate += datetime.timedelta(days=1)
                    if cur_date not in news_by_date:
                        return None
                    for i in news_by_date[cur_date]:
                        word_num.append(news_by_id[i])
                    wlen = len(word_num)
                    if wlen > d * length:
                        word_num = word_num[0: d*length]
                    elif wlen < d * length:
                        for i in range(wlen,d*length):
                            word_num.append(word_num_append)
                
                data = []
                for k in date_stock['stock_id'].keys():
                    data.append([word_num, [str(np.round(float(history['rate'][k]), 2)), str(date_stock['date'][k]), str(date_stock['stock_id'][k])]])                

                print ("正在处理的日期:%s"%date_stock.name)            
                pd.DataFrame(data).to_csv(output_path, index=False, header=False, mode="a", encoding="utf-8")

            g_stock = history.groupby(by = ["date"])                    
            cls.groupby_skip = False
            g_stock.apply(func_train_data)
            slimit += skip
    """