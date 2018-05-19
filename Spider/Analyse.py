#coding=utf-8
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

class Analyse(object):
    groupby_skip = False

    @classmethod
    def getStockRateDistributed(cls, cur=None, stock_ids_str=None, start_date=None, end_date=None):
        if cur == None or stock_ids_str == None or start_date == None or end_date == None:
            return None

        cur.execute("SELECT stock_id, closing, date FROM history WHERE stock_id in (%s) and date between '%s' and '%s' order by stock_id, date " % (stock_ids_str, start_date, end_date))
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

        x_1 = []
        x_2 = []
        for i in range(-10, 1):
            x_1.append(i)
        for i in range(0, 11):
            x_2.append(i)

        y_stock = {}
        def func_train_data(date_stock):
            if cls.groupby_skip == False:
                cls.groupby_skip = True
                return None
            
            y = []
            x_1_length = len(x_1)
            x_2_length = len(x_2)
            for i in range(x_1_length+x_2_length):
                y.append(0)

            for k in date_stock['stock_id'].keys():
                rate = date_stock['rate'][k]
                for k in range(len(x_1)):
                    if x_1[k] == 0 and rate < x_1[k]:
                        y[k] += 1
                    elif rate <= x_1[k]:
                        y[k] += 1

                for k in range(len(x_2)):
                    if rate >= x_2[k]:
                        y[x_1_length+k] += 1
            y_stock[date_stock.name] = y
            print ("正在处理的股票ID:%s"%date_stock.name)

        g_stock = history.groupby(by = ["stock_id"])
        cls.groupby_skip = False
        g_stock.apply(func_train_data)

        x = x_1 + x_2
        fields = [
            ['red', 'stock id '],
            ['blue', 'stock id '],
            ['yellow', 'stock id ']
        ]

        for k in y_stock:
            all_num = y_stock[k][10] + y_stock[k][11]
            for i in range(len(x)):
                y_stock[k][i] = float('%.2f' % (float(y_stock[k][i]) / all_num))

        # 画图 以折线图表示结果
        plt.figure()
        i = 0
        print(x)
        for k in y_stock:
            print(y_stock[k])
            plt.plot(x, y_stock[k], color=fields[i][0], label='%s %s' % (fields[i][1], k))
            i += 1
            for a, b in zip(x, y_stock[k]):
                plt.text(a, b, b, ha='center', va='bottom', fontsize=7)

        plt.legend() # 显示图例
        plt.title('rate distributed')
        plt.xlabel('rate division')
        plt.ylabel('percent')
        plt.show()
    
    @classmethod
    def getStockPreDistributed(cls, stock_name='', basic_path=None, folder=None, file=None):
        if folder is None or file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))

        test_part_array = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        test_part_times = 10

        # test_part_array = [0, 0.1]
        # test_part_times = 1

        division = range(-10, 11, 1)
        x = []
        y_stock_data = []
        for i in range(len(division)+1):
            y_stock_data.append(0)
            if i == 0:
                x.append('~%s' % division[i])
            elif i == len(division):
                x.append('%s~' % division[i-1])
            else:
                x.append('%s~%s' % (division[i-1], division[i]))
        
        up = 0
        down = 0
        profit = 0
        is_handle = False
        y_stock_predictions = []

        train_num = 0
        test_num = 0

        for i in range(len(test_part_array)-1):
            for j in range(test_part_times):
                file_path = os.path.join(basic_path, folder + '_%s_%s'%(i, j), file)
                data_temp = pd.read_csv(file_path)
                                
                rates = data_temp['rate'].tolist()
                predictions = data_temp['predictions'].apply(eval).tolist()

                all_len = len(data_temp['rate'])
                start = int(all_len*test_part_array[i])
                end = int(all_len*test_part_array[i+1])                
                
                train_rates = rates[:start] + rates[end:]
                train_predictions = predictions[:start] + predictions[end:]                

                test_rates = rates[start: end]
                test_predictions = predictions[start: end]

                if len(y_stock_predictions) == 0:
                    for k in range(len(test_predictions[0])):
                        y_stock_predictions.append({'pre': copy.deepcopy(y_stock_data), 'train_acc': 0, 'test_acc': 0, 'pro': 0})            
                                
                # train
                for k in range(len(train_rates)):
                    rate = train_rates[k]
                    
                    if not is_handle:
                        if rate > 0:
                            up += 1
                        else:
                            down += 1
                        profit += rate

                    for p_k in range(len(train_predictions[k])):
                        if (rate <= 0 and train_predictions[k][p_k] == 0) or (rate > 0 and predictions[k][p_k] == 1):
                            y_stock_predictions[p_k]['train_acc'] += 1

                # test
                for k in range(len(test_rates)):
                    rate = test_rates[k]

                    if not is_handle:
                        if rate > 0:
                            up += 1
                        else:
                            down += 1
                        profit += rate

                    for d_k in range(len(division)+1):
                        if (d_k == 0 and rate <= division[d_k]) or (d_k == len(division) and rate > division[d_k-1]) or (rate > division[d_k-1] and rate <= division[d_k]):
                            y_stock_data[d_k] += 1
                            break

                    for p_k in range(len(test_predictions[k])):
                        if (rate <= 0 and test_predictions[k][p_k] == 0) or (rate > 0 and test_predictions[k][p_k] == 1):
                            y_stock_predictions[p_k]['pre'][d_k] += 1
                            y_stock_predictions[p_k]['test_acc'] += 1

                        if test_predictions[k][p_k] == 1:
                            y_stock_predictions[p_k]['pro'] += rate

                if not is_handle:
                    train_num += (all_len - (end - start))
                    test_num += (end - start)
                    is_handle = True

        for i in range(len(y_stock_data)):
            y_stock_data[i] /= test_part_times        

        num = (len(test_part_array)-1) * test_part_times
        y_stock_predictions_per = copy.deepcopy(y_stock_predictions)
        for i in range(len(y_stock_predictions)):
            for j in range(len(y_stock_predictions[i]['pre'])):
                y_stock_predictions[i]['pre'][j] /= test_part_times
                y_stock_predictions_per[i]['pre'][j] = float('%.2f' % (y_stock_predictions[i]['pre'][j] / y_stock_data[j])) if y_stock_data[j] != 0 else 0

            y_stock_predictions[i]['train_acc'] /= num
            y_stock_predictions[i]['test_acc'] /= num
            y_stock_predictions[i]['pro'] /= test_part_times

            y_stock_predictions_per[i]['train_acc'] = float('%.4f' % (y_stock_predictions[i]['train_acc'] / train_num))
            y_stock_predictions_per[i]['test_acc'] = float('%.4f' % (y_stock_predictions[i]['test_acc'] / test_num))
            y_stock_predictions_per[i]['pro'] = float('%.2f' % y_stock_predictions[i]['pro'])
        
        fields = [
            ['green', 'TW-CNN'],
            ['purple', 'BW-CNN'],
            ['yellow', 'BI-CNN'],
            ['red', 'BM&S-CNN'],
            ['pink', 'S-LSTM'],
            ['gray', 'BI&S-LSTM'],
            ['burlywood', 'BM&S-LSTM'],
            ['dimgray', 'EL-MODEL']            
        ]

        up_percent = float('%.4f' % (float(up) / (up + down)))
        down_percent = float('%.4f' % (float(down) / (up + down)))
        print('up: %s, down: %s, profit: %.2f' % (up_percent, down_percent, profit))    
        for i in range(len(y_stock_predictions_per)):
            print('%s : train_accuracy %s, test_accuracy %s, pro %s' % (fields[i][1], y_stock_predictions_per[i]['train_acc'], y_stock_predictions_per[i]['test_acc'], y_stock_predictions_per[i]['pro']))

        # 画图 以折线图表示结果
        plt.figure('fig1')
        plt.title('predictions distributed 1')
        x_range = range(len(y_stock_data))
        plt.plot(x_range, y_stock_data, color='blue', label='Stock')        
        for i in range(len(y_stock_predictions)):
            plt.plot(x_range, y_stock_predictions[i]['pre'], color=fields[i][0], label=fields[i][1])
        plt.xticks(x_range, x, rotation=45)

        plt.legend() # 显示图例
        plt.xlabel('rate division')
        plt.ylabel('numbers')
        plt.tick_params(labelsize=6)

        # 画图 以折线图表示结果
        plt.figure('fig2')
        plt.title('predictions distributed')
        x_range = range(len(y_stock_data))
        for i in range(len(y_stock_predictions_per)):
            plt.plot(x_range, y_stock_predictions_per[i]['pre'], color=fields[i][0], label=fields[i][1])
        plt.xticks(x_range, x, rotation=45)

        plt.legend() # 显示图例
        plt.xlabel('rate division')
        plt.ylabel('percent')
        plt.tick_params(labelsize=6)
        plt.show()