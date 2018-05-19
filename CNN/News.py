#coding=utf-8
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import jieba.posseg as pseg
import os
import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Spider.tool import VTool

import matplotlib.pyplot as plt

class News(object):
    def getHighRateDate(self, cur=None, stock_id_str=None, start_date=None, end_date=None, rate=2):
        if cur == None or stock_id_str == None or start_date == None or end_date == None or rate < 0:
            return None        

        cur.execute("SELECT stock_id, closing, date FROM history WHERE stock_id in (%s) and date between '%s' and '%s' order by stock_id, date " % (stock_id_str, start_date, end_date))
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

        left_shift = 2
        test_part_array = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        lens = len(history['date'])
        
        high_rate_dates = []
        for i in range(len(test_part_array)-1):
            index = 0
            start = int(lens*test_part_array[i])
            end = int(lens*test_part_array[i+1])
            for k in history['rate'].keys():
                if index == start:
                    start = str(history['date'][k])
                if index == end:
                    end = str(history['date'][k])
                    break
                index += 1

            stocks = {}
            news_date = {}
            for k in history['rate'].keys():
                date = str(history['date'][k])
                if date not in stocks and (date < str(start) or date > str(end)) and (k-left_shift) in history['date']:
                    stocks[date] = {}
                    news_date[date] = str(history['date'][k-left_shift])
                else:
                    continue
                stocks[date][history['stock_id'][k]] = history['rate'][k]                    

            high_rate_date = {"up":[], "down":[]}
            for d in stocks:
                if len(stocks[d]) == 0:
                    continue
                up = down = True
                for k in stocks[d]:
                    if stocks[d][k] >= 0:
                        down = False
                        if stocks[d][k] < rate:
                            up = False
                    else:
                        up = False
                        if stocks[d][k] > -1*rate:
                            down = False
                if up == True:
                    high_rate_date['up'].append([str(d), news_date[d], float(str(np.round(float(stocks[d][k]), 2)))])
                if down == True:
                    high_rate_date['down'].append([str(d), news_date[d], float(str(np.round(float(stocks[d][k]), 2)))])
            high_rate_dates.append(high_rate_date)

        return high_rate_dates
    
    def jiebafenci(self, all_the_text):
        re = []
        flag_list = ['t','q','p','u','e','y','o','w','m']
        words = pseg.cut(all_the_text)
        for w in words:
            flag = w.flag
            tmp = w.word
            #print "org: "+tmp
            if len(tmp)>1 and len(flag)>0 and flag[0] not in flag_list and  tmp[0]>=u'/u4e00' and tmp[0]<=u'\u9fa5':
                re.append(w.word)
        return re

    def calcuWordTrend(self, cur=None, choose_dates=None, basic_path=None, word_cache_file=None, output_file=None):
        if cur == None or choose_dates == None or output_file == None or word_cache_file == None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        word_cache_path = os.path.join(basic_path, word_cache_file)

        index = 0
        is_reload = False
        for choose_date in choose_dates:
            output_path = os.path.join(basic_path, output_file + '_%s.csv' % index)

            index += 1            
            if os.path.exists(output_path) and not is_reload:
                continue
            
            VTool.makeDirs(files=[output_path])

            date_str_arr = []
            date_rate = {}
            for k in choose_date:
                for d in choose_date[k]:
                    date = d[1]
                    date_str_arr.append('"'+date+'"')
                    date_rate[date] = d[2]
            date_str = ",".join(date_str_arr)        

            news = []
            if len(date_str_arr) > 0:
                cur.execute("SELECT id, time FROM news WHERE time in (%s) order by time, content" % (date_str))
                news_temp = cur.fetchall()

                news_by_id = {}
                for n in news_temp:
                    news_by_id[n[0]] = {}
                    news_by_id[n[0]]['date'] = str(n[1])
                    news_by_id[n[0]]['words'] = ''
                del news_temp
                
                nid_len = len(news_by_id)
                reader = pd.read_csv(word_cache_path, chunksize=1000)
                for sentences in reader:
                    for k in sentences['1'].keys():
                        nid = sentences['0'][k]
                        if nid in news_by_id and news_by_id[nid]['words'] == '':
                            news_by_id[nid]['words'] = str(sentences['1'][k]).split(" ")
                            nid_len-=1
                    if nid_len <= 0:
                        break
                reader.close()
                del reader, sentences
            print(len(news_by_id))
            
            word_dict = {"words":{}, "up_total_words":0, "down_total_words":0}
            i = 0
            for k in news_by_id:
                date = news_by_id[k]['date']
                if date not in date_rate:                
                    continue            
                if date_rate[date] >= 0:
                    ckey = "up"
                else:
                    ckey = "down"
                words = news_by_id[k]['words']
                for w in words:
                    if w not in word_dict["words"]:
                        word_dict["words"][w] = {"up":0 ,"down":0}
                    word_dict["words"][w][ckey] += 1
                    word_dict["%s_total_words" % ckey] += 1

                i += 1
                print(i)

            if word_dict["up_total_words"] != 0:
                for w in word_dict["words"]:
                    word_dict["words"][w]["up"] = word_dict["words"][w]["up"] / word_dict["up_total_words"]
            if word_dict["down_total_words"] != 0:
                for w in word_dict["words"]:
                    word_dict["words"][w]["down"] = word_dict["words"][w]["down"] / word_dict["down_total_words"]

            csv_data = []
            for w in word_dict["words"]:
                csv_data.append([w, word_dict["words"][w]["up"], word_dict["words"][w]["down"]])
            csv_data.append(['total_words', word_dict["up_total_words"], word_dict["down_total_words"]])

            pd.DataFrame({"0":[], "1":[], "2":[]}).to_csv(output_path, index=False, encoding="utf-8")
            pd.DataFrame(csv_data).to_csv(output_path, index=False, header=False, mode="a", encoding="utf-8")

    def wordTrendTest(self, cur=None, basic_path=None, word_trend_files=None, stock_id=None, start_date=None, end_date=None, rate=3):
        if cur == None or word_trend_files == None or stock_id == None or start_date == None or end_date == None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        for k in range(len(word_trend_files)):
            word_trend_files[k] = os.path.join(basic_path, word_trend_files[k])

        output_path = os.path.join(basic_path, word_trend_files[0] + '.test')

        if not os.path.exists(output_path):
            cur.execute("SELECT stock_id, percentage_difference, date FROM history WHERE stock_id in (%s) and date between '%s' and '%s' order by stock_id, date " % (stock_id, start_date, end_date))
            history_t = cur.fetchall()

            history_temp = []
            for h in zip(*history_t):
                history_temp.append(h)
            history = {'stock_id':history_temp[0], 'percentage_difference':history_temp[1], 'date':history_temp[2]}
            del history_t, history_temp
            history = DataFrame(history)
            g_history = history.groupby(by = ['stock_id'])
            history.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

            left_shift = 2
            stocks = {}
            for k in history['percentage_difference'].keys():
                date = str(history['date'][k])
                if date not in stocks and (k-left_shift) in history['date'] and (history['percentage_difference'][k] >= rate or history['percentage_difference'][k] <= -rate):
                    stocks[date] = [str(history['date'][k-left_shift]), history['percentage_difference'][k]]
            
            date_str_arr = []
            date_rate = {}
            for k in stocks:                
                date_str_arr.append('"%s"'%stocks[k][0])
                date_rate[stocks[k][0]] = stocks[k][1]
            date_str = ",".join(date_str_arr)

            cur.execute("SELECT id, content FROM news WHERE time in (%s) order by time, content" % (date_str))
            news_temp = cur.fetchall()
            news = {}
            for n in news_temp:
                news[str(n[0])] = n[1]
            del news_temp        

            cur.execute("SELECT GROUP_CONCAT(id  SEPARATOR ','), time FROM news WHERE time in (%s) group by time" % (date_str))
            news_time = cur.fetchall()
            
            ys = []
            for f in word_trend_files:
                word_trend = {}
                word_trend_temp = pd.read_csv(f)

                for k in word_trend_temp["0"].keys():
                    word_trend[word_trend_temp["0"][k]] = [word_trend_temp["1"][k], word_trend_temp["2"][k]]
                if int(word_trend['total_words'][0]) == 0 or int(word_trend['total_words'][1]) == 0:
                    return None
                p_up = word_trend['total_words'][0] / (word_trend['total_words'][0] + word_trend['total_words'][1])
                p_down = word_trend['total_words'][1] / (word_trend['total_words'][0] + word_trend['total_words'][1])

                date_trend = {}
                i = 0
                print(len(news_time))
                for nt in news_time:
                    date = str(nt[1])                    
                    date_trend[date] = 0
                    if date not in date_rate:
                        continue
                    sids = nt[0].split(",")
                    for sid in sids:
                        sid = str(sid)
                        if sid not in news:                    
                            continue
                        words = self.jiebafenci(news[sid])
                        wp_up = p_up
                        wp_down = p_down
                        for w in words:
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
                        
                        date_trend[date] += wp_up / (wp_up + wp_down)                
                    if len(sids) != 0 :
                        date_trend[date] /= len(sids)

                    i += 1
                    print(i)

                data = []
                for k in date_trend:
                    data.append([k, float(date_rate[k]), float('%.2f'%date_trend[k]), float('%.2f'%(1-float('%.2f'%date_trend[k])))])
                ys.append(data)

            data = ys
            pd.DataFrame(data).to_csv(output_path, index=False, mode="a", encoding="utf-8")
        
        else:
            data = pd.read_csv(output_path)
            word_trend_num = 0
            for i in data:
                word_trend_num = len(data[i])
                data[i] = data[i].apply(eval)
            data_temp = []
            for k in range(word_trend_num):
                data_temp.append([])        
            for i in data:
                for j in range(word_trend_num):
                    data_temp[j].append(data[i][j])
            data = data_temp

        fields = [
            ['red', 'Rate '],
            ['green', 'Word trend '],
            ['purple', 'Word trend '],
            ['yellow', 'Word trend '],
        ]

        x = range(len(data[0]))        
        y = []
        for k in range(len(data)+1):
            y.append([])
        
        for d in data[0]:
            y[0].append(d[1])

        index = 0
        for d in data:
            for dd in d:
                y[index+1].append(float('%.2f'% ((dd[2]-dd[3])*10)))
            index += 1
        
        # 画图 以折线图表示结果
        plt.figure('fig1')
        plt.title('word trend & rate')

        print(x)
        print(y)
        index = 0
        for k in range(len(y)):
            if index == 0:
                plt.plot(x, y[k], color=fields[k][0], label=fields[k][1])
            else:
                plt.plot(x, y[k], color=fields[k][0], label=fields[k][1] + str(index))
            index += 1

        plt.legend() # 显示图例
        plt.xlabel('date')
        plt.ylabel('data')
        plt.tick_params(labelsize=6)
        plt.show()