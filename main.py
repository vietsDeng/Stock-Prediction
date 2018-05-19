#coding=utf-8

from Database.VMysql import VMysql

from Spider.StockSpider import StockSpider
from Spider.NewsSpider import NewsSpider
from Spider.tool import VTool
from FetchBindex.BindexSpider import BindexSpider

from CNN.OriginData import OriginData
from CNN.DataHelper import DataHelper
from CNN.News import News

from CNN.CNNStockText import CNNStockText
from CNN.CNNStockNumber import CNNStockNumber

from LSTM.LSTMStockOrigin import LSTMStockOrigin
from LSTM.OriginData import OriginData as LSTMOriginData

from Ensemble.Ensemble import Ensemble

from Spider.tool import VTool
from Spider.Analyse import Analyse

import os
import numpy as np
import pandas as pd

conn, cur = VMysql.get_mysql()
basic_path = 'D:\\workspace\\Mine\\run_data'

"""
Spider
"""
# stockSpider = StockSpider()
# fields = [['600104', 'cn', '上汽集团'], 
#             ['601318', 'cn', '中国平安'],
#             ['002230', 'cn', '科大讯飞']]
# stockSpider.craw(conn=conn, cur=cur, fields=fields)

# newsSpider = NewsSpider()
# newsSpider.craw(conn=conn, cur=cur)

# bindexSpider = BindexSpider()
# bindexSpider.craw(conn=conn, cur=cur)


# DataHelper.make_word2vec_model(basic_path, "title_keyword_cache.csv", file_to_save = 'news_title_word2vec', embedding_size = 64, step = 1, min_count = 3)
# DataHelper.make_word2vec_model(basic_path, "tfidf_keyword_cache.csv", file_to_save = 'news_tfidf_word2vec', embedding_size = 64, step = 1, min_count = 3)

'''
CNN_DATA
'''

folder_600104 = "600104/" #3
folder_601318 = "601318/" #5
folder_002230 = "002230/" #7

"""
1.制作标题-分类数据
"""
output_file = "cnn/title_data.csv"
# OriginData.makeTitleOriginCsv(cur=cur, start_date='2010-01-01', end_date='2017-12-01', day_num=1, stock_id="7", basic_path=basic_path, input_file='title_keyword_cache.csv', output_file=folder_002230+output_file)

"""
新闻关键词各种处理
"""
# OriginData.makeNewsKeywordCacheCsv(cur=cur, start_date='2017-01-17', end_date='2017-12-01', analyse_type='tfidf', rewrite=False, basic_path=basic_path)
# OriginData.makeNewsKeywordCacheCsv(cur=cur, start_date='2017-02-17', end_date='2017-12-01', analyse_type='all', rewrite=False, basic_path=basic_path)
# OriginData.makeNewsKeywordCacheCsv(cur=cur, start_date='2010-01-01', end_date='2017-12-01', basic_path=basic_path, analyse_type='title', rewrite=True)

# OriginData.makeImportVocab(keyword_csv_file="tfidf_keyword_cache.csv", important_vocab_csv_file="tfidf_important_vocab.csv", basic_path=basic_path)

# Origin.importWordToMysql(cur=cur, tfidf_file="tfidf_important_vocab.csv", textrank_file="textrank_important_vocab.csv", word_count=100)
# OriginData.importCityToMysql(cur=cur)

"""
2.制作百度指数-分类数据
"""
output_file = "cnn/bindex_data.csv"
# words = OriginData.getImportVocab(cur, count=50)
# OriginData.makeBindexOriginCsv(cur=cur, words=words, start_date='2010-01-01', end_date='2017-12-01', day_num=1, stock_id="7", basic_path=basic_path, output_file=folder_002230+output_file)

"""
3.制作新闻关键词-分类数据
"""
# OriginData.makeTextOriginCsv(cur=cur, start_date='2010-01-01', end_date='2017-12-01', day_num=1, basic_path=basic_path, input_file='tfidf_keyword_cache.csv', output_file=folder_002230+'cnn/tfidf_text_data.csv', stock_id="7", rewrite=True)

"""
4.制作新闻趋势历史数据-分类数据
"""
# news = News()
# rate_dates = news.getHighRateDate(cur=cur, stock_id_str="7", start_date="2010-01-01", end_date="2017-12-01", rate=4)
# news.calcuWordTrend(cur=cur, choose_dates=rate_dates, basic_path=basic_path, word_cache_file="all_keyword_cache.csv", output_file=folder_002230+"cnn/word_trend")

# 
# for i in range(10):
#     OriginData.makeTrendStockOriginCsv(cur=cur, start_date="2010-01-01", end_date="2017-12-01", day_num=20, stock_id="7", basic_path=basic_path, output_file=folder_002230+"cnn/news_stock_data_%s.csv"%i, word_trend_file=folder_002230+"cnn/word_trend_%s.csv"%i, news_file="all_keyword_cache.csv")

##
# 模型预测
##
    
"""
CNN_MODEL
"""
input_file = "cnn/title_data.csv"
# cst = CNNStockText()
# cst.train(basic_path=basic_path, input_file=folder_600104+input_file, output_folder=folder_600104+"cnn/title_run", word2vec_model="news_title_word2vec", test_part_start=0.9, test_part_end=1, times=1)
# cst.predict(basic_path=basic_path, input_file=folder_600104+input_file, output_folder=folder_600104+"cnn/title_run", word2vec_model="news_title_word2vec", test_part_start=0.9, test_part_end=1)

input_file = "cnn/tfidf_text_data.csv"
# cst = CNNStockText()
# cst.train(basic_path=basic_path, input_file=folder_600104+input_file, output_folder=folder_600104+"cnn/text_run", word2vec_model="news_tfidf_word2vec", test_part_start=0.9, test_part_end=1, times=1)
# cst.predict(basic_path=basic_path, input_file=folder_600104+input_file, output_folder=folder_600104+"cnn/text_run", word2vec_model="news_tfidf_word2vec", test_part_start=0.9, test_part_end=1)

####

input_file = "cnn/bindex_data.csv"
# csn = CNNStockNumber()
# csn.train(basic_path=basic_path, input_file=folder_600104+input_file, output_folder=folder_600104+"cnn/bindex_run", embedding_dim=3, test_part_start=0.9, test_part_end=1, times=10)
# csn.predict(basic_path=basic_path, input_file=folder_600104+input_file, output_folder=folder_600104+"cnn/bindex_run", embedding_dim=3, test_part_start=0.9, test_part_end=1)

input_file = "cnn/news_stock_data.csv"
# csn = CNNStockNumber()
# csn.train(basic_path=basic_path, input_file=folder_600104+input_file, output_folder=folder_600104+"cnn/news_stock_run", embedding_dim=10, test_part_start=0.9, test_part_end=1, times=30)
# csn.predict(basic_path=basic_path, input_file=folder_600104+input_file, output_folder=folder_600104+"cnn/news_stock_run", embedding_dim=10, test_part_start=0.9, test_part_end=1)

'''
LSTM_MODEL
'''
"""
1
"""
output_file = "lstm/stock_origin_data.csv"
rate_model_folder = "lstm/origin_model"

# LSTMOriginData.makeOriginDataCsv(cur=cur, start_date='2010-01-01', end_date='2017-12-01', basic_path=basic_path, output_file=folder_002230+output_file, stock_id="7")

# so = LSTMStockOrigin()
# so.train_rate(basic_path=basic_path, data_file=folder_600104+output_file, model_folder=folder_600104+rate_model_folder, input_type="origin", word_count=0, input_size=8, batch_size=20, time_step=10, test_part_start=0.9, test_part_end=1, times=300)
# so.predict_rate(basic_path=basic_path, data_file=folder_600104+output_file, model_folder=folder_600104+rate_model_folder, test_part_start=0.9, test_part_end=1)


"""
2
"""
output_file = 'lstm/stock_bindex_data.csv'
rate_model_folder = "lstm/bindex_model"

# LSTMOriginData.makeBindexDataCsv(cur=cur, start_date='2010-01-01', end_date='2017-12-01', basic_path=basic_path, output_file=folder_002230+output_file, stock_id="7")

# so = LSTMStockOrigin()
# so.train_rate(basic_path=basic_path, data_file=folder_600104+output_file, model_folder=folder_600104+rate_model_folder, input_type="bindex", word_count=20, input_size=28, batch_size=30, time_step=10, test_part_start=0.9, test_part_end=1, times=300)
# so.predict_rate(basic_path=basic_path, data_file==folder_600104+output_file, model_folder=folder_600104+rate_model_folder, test_part_start=0.9, test_part_end=1)

"""
3
"""
output_file = 'lstm/stock_news_data.csv'
rate_model_folder = "lstm/news_model"

# for i in range(10):
#     LSTMOriginData.makeNewsDataCsv(cur=cur, start_date='2010-01-01', end_date='2017-12-01', basic_path=basic_path, news_file="all_keyword_cache.csv", word_trend_file=folder_002230+"cnn/word_trend_%s.csv"%i, output_file=folder_002230+'lstm/stock_news_data_%s.csv'%i, stock_id="7")

# so = LSTMStockOrigin()
# so.train_rate(basic_path=basic_path, data_file=folder_600104+output_file, model_folder=folder_600104+rate_model_folder, input_type="news", word_count=0, input_size=10, batch_size=2, time_step=10, test_part_start=0.9, test_part_end=1, times=1)
# so.predict_rate(basic_path=basic_path, data_file=folder_600104+output_file, model_folder=folder_600104+rate_model_folder, test_part_start=0.9, test_part_end=1)

##
# 十折十次交叉验证
##
if False:
    test_part_array = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    test_part_times = 10
    is_reload = False
    is_train = True

    folder_600104 = "600104/" #3
    folder_601318 = "601318/" #5
    folder_002230 = "002230/" #7

    choose_stock_folders = [folder_601318, folder_002230]
    choose_models = [8]

    for choose_stock_folder in choose_stock_folders:
        for choose_model in choose_models:
        
            if choose_model == 1:

                model_folder = "cnn/title_run"
                input_file = "cnn/title_data.csv"
                word2vec_model = "news_title_word2vec"
                reduce_num = 21
                filter_sizes = [3, 4, 5]
                times = 3

            elif choose_model == 2:

                model_folder = "cnn/text_run"
                input_file = "cnn/tfidf_text_data.csv"    
                word2vec_model = "news_tfidf_word2vec"
                reduce_num = 21
                filter_sizes = [8, 9, 10]
                times = 3

            elif choose_model == 3:
                
                model_folder = "cnn/bindex_run"
                input_file = "cnn/bindex_data.csv"
                embedding_dim = 3
                reduce_num = 21
                filter_sizes = [2]
                stand = False

            elif choose_model == 4:
                
                model_folder = "cnn/news_stock_run"
                input_file = "cnn/news_stock_data.csv"
                embedding_dim = 10                
                reduce_num = 0
                filter_sizes = [3, 4, 5]
                stand = True

            elif choose_model == 5:

                input_file = "lstm/stock_origin_data.csv"
                model_folder = "lstm/origin_model"
                input_type = "origin"
                word_count = 0
                input_size = 8
                reduce_num = 10

            elif choose_model == 6:

                input_file = "lstm/stock_bindex_data.csv"
                model_folder = "lstm/bindex_model"
                input_type = "bindex"
                word_count = 20
                input_size = 8
                reduce_num = 10

            elif choose_model == 7:

                input_file = "lstm/stock_news_data.csv"
                model_folder = "lstm/news_model"
                input_type = "news"
                word_count = 0
                input_size = 10
                reduce_num = 10

            elif choose_model == 8:

                model_folder = "ensemble"
                input_file = "ensemble_data.csv"                
                reduce_num = 0

            else:
                exit()

            res_file = os.path.join(basic_path, choose_stock_folder, model_folder, "ten-fold-ten-times.csv")

            # 读入情况
            columns = []
            for i in range(len(test_part_array)-1):
                columns.append(str(i))

            csv_res = {}
            if is_reload or not os.path.exists(res_file):
                for i in range(len(test_part_array)-1):
                    csv_res[str(i)] = []
                    for j in range(test_part_times):
                        csv_res[str(i)].append([])
                VTool.makeDirs(files=[res_file])
                pd.DataFrame(csv_res).to_csv(res_file, index=False, columns=columns)
            csv_res = pd.read_csv(res_file)

            array_start = len(test_part_array) - 1
            times_start = test_part_times
            res = {}
            find = False
            for i in csv_res:
                res[i] = csv_res[i].apply(eval).values
                if not find:
                    for j in range(len(res[i])):
                        if len(res[i][j]) == 0:
                            array_start = int(i)
                            times_start = int(j)
                            find = True
                            break

            for i in range(array_start, len(test_part_array)-1):
                for j in range(times_start, test_part_times):
                    
                    folder_extra = '_' + str(i) + '_' + str(j)

                    if choose_model == 1 or choose_model == 2:

                        cst = CNNStockText()
                        is_train and cst.train(basic_path=basic_path, input_file=choose_stock_folder+input_file, output_folder=choose_stock_folder+model_folder, word2vec_model=word2vec_model, filter_sizes=filter_sizes, folder_extra=folder_extra, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1], times=times)
                        accuracy, profit, origin_profit, predictions, _ = cst.predict(basic_path=basic_path, input_file=choose_stock_folder+input_file, output_folder=choose_stock_folder+model_folder, word2vec_model=word2vec_model, filter_sizes=filter_sizes, folder_extra=folder_extra, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1])
                        num = len(predictions)

                    elif choose_model == 3 or choose_model == 4:

                        if choose_model == 4:                            
                            input_file = "cnn/news_stock_data_%s.csv" % i

                        csn = CNNStockNumber()
                        is_train and csn.train(basic_path=basic_path, input_file=choose_stock_folder+input_file, output_folder=choose_stock_folder+model_folder, embedding_dim=embedding_dim, folder_extra=folder_extra, reduce_num=reduce_num, filter_sizes=filter_sizes, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1], data_stand=stand, times=20)
                        accuracy, profit, origin_profit, predictions, _ = csn.predict(basic_path=basic_path, input_file=choose_stock_folder+input_file, output_folder=choose_stock_folder+model_folder, embedding_dim=embedding_dim, folder_extra=folder_extra, filter_sizes=filter_sizes, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1], data_stand=stand)
                        num = len(predictions)

                    elif choose_model == 5:
                        
                        so = LSTMStockOrigin()
                        is_train and so.train_rate(basic_path=basic_path, data_file=choose_stock_folder+input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, input_type=input_type, word_count=word_count, input_size=input_size, batch_size=20, time_step=10, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1], times=30)
                        accuracy, profit, origin_profit, predictions, _ = so.predict_rate(basic_path=basic_path, data_file=choose_stock_folder+input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1])
                        num = len(predictions)
                        
                    elif choose_model == 6:
                        
                        so = LSTMStockOrigin()
                        is_train and so.train_rate(basic_path=basic_path, data_file=choose_stock_folder+input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, input_type=input_type, word_count=word_count, input_size=input_size, batch_size=20, time_step=10, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1], times=30)
                        accuracy, profit, origin_profit, predictions, _ = so.predict_rate(basic_path=basic_path, data_file=choose_stock_folder+input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1])
                        num = len(predictions)

                    elif choose_model == 7:
                        input_file = "lstm/stock_news_data_%s.csv" % i
                        
                        so = LSTMStockOrigin()
                        is_train and so.train_rate(basic_path=basic_path, data_file=choose_stock_folder+input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, input_type=input_type, word_count=word_count, input_size=input_size, batch_size=20, time_step=10, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1], times=30)
                        accuracy, profit, origin_profit, predictions, _ = so.predict_rate(basic_path=basic_path, data_file=choose_stock_folder+input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1])
                        num = len(predictions)

                    elif choose_model == 8:
                        
                        ens = Ensemble()
                        is_train and ens.train(basic_path=basic_path, input_file=input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, batch_size=10, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1], times=1)
                        accuracy, profit, origin_profit, predictions, _ = ens.predict(basic_path=basic_path, input_file=input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, reduce_num=reduce_num, test_part_start=test_part_array[i], test_part_end=test_part_array[i+1])
                        num = len(predictions)

                        ens.predictAndMake(basic_path=basic_path, input_file=input_file, model_folder=choose_stock_folder+model_folder, folder_extra=folder_extra, output_file='all_predictions.csv')

                    else:
                        exit()
                    
                    print(folder_extra)
                    # 写配置
                    res[str(i)][j] = ["%.2f" % (accuracy*100), "%.2f" % profit, "%.2f" % origin_profit, "%d" % num]
                    pd.DataFrame(res).to_csv(res_file, index=False, columns=columns)   
                times_start = 0

# ens = Ensemble()
# ens.makeEnsembleData(stock_folders=[folder_002230])

###
# 处理数据
###
if True:
    folder_600104 = "600104/" #3
    folder_601318 = "601318/" #5
    folder_002230 = "002230/" #7

    stock_folders = [folder_002230]

    predict_res_csvs = [
        'cnn/title_run/ten-fold-ten-times.csv',
        'cnn/text_run/ten-fold-ten-times.csv',
        'cnn/bindex_run/ten-fold-ten-times.csv',
        'cnn/news_stock_run/ten-fold-ten-times.csv',
        'lstm/origin_model/ten-fold-ten-times.csv',
        'lstm/bindex_model/ten-fold-ten-times.csv',
        'lstm/news_model/ten-fold-ten-times.csv',
        'ensemble/ten-fold-ten-times.csv'
    ]

    for choose_stock_folder in stock_folders:
        for predict_res_csv in predict_res_csvs:        

            res_file = os.path.join(basic_path, choose_stock_folder, predict_res_csv)
            # 读入情况
            csv_res = pd.read_csv(res_file)

            all_accuracy, all_num, profit, origin_profit = 0, 0, 0, 0
            for column in csv_res.columns:
                csv_res[column] = csv_res[column].apply(eval)
                temp_accuracy, temp_profit, temp_origin_profit = 0, 0, 0
                for res in csv_res[column]:
                    temp_accuracy += float(res[0])
                    temp_profit += float(res[1])
                    temp_origin_profit += float(res[2])

                l = len(csv_res[column])
                all_accuracy += temp_accuracy / l * float(csv_res[column][0][3])
                all_num += int(csv_res[column][0][3])
                profit += temp_profit / l
                origin_profit += temp_origin_profit / l
            accuracy = all_accuracy / all_num

            print("%s\naccuracy: %.2f, profit: %.2f, origin_profit: %.2f\n" % (predict_res_csv, accuracy, profit, origin_profit))            

# news = News()

# news.wordTrendTest(cur=cur, basic_path=basic_path, word_trend_files=[folder_600104+"cnn/word_trend_0.csv", folder_600104+"cnn/word_trend_1.csv", folder_600104+"cnn/word_trend_2.csv"], stock_id='3', start_date='2010-01-01', end_date='2010-10-01', rate=3)

# Analyse.getStockRateDistributed(cur=cur, stock_ids_str='3', start_date='2010-01-01', end_date='2017-12-01')
Analyse.getStockPreDistributed(basic_path=basic_path+'\\'+folder_002230+'ensemble', folder='checkpoints', file='all_predictions.csv')


##################

# cur.close()
# conn.close()