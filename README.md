### 《基于深度学习的财经新闻量化与股市预测研究》论文源代码说明
---


#### 目录说明
- CNN (CNN文本处理及模型)
- Database (mysql连接类及数据库结构文件)
- Ensemble (集成学习文本处理及模型)
- FetchBindex (百度指数爬虫)
- LSTM (LSTM文本处理及模型)
- run_data (生成文件存放目录)
- Spider (财经新闻、股票数据爬虫及自定义工具类)
- /main.py (入口文件，含交叉验证及所有样例)
- /ensemble_temp.bat (批处理文件，作用详见下面"缺陷")

#### 数据爬取
1. 财经新闻数据：Spider/NewsSpider.py
2. 股票历史数据：Spider/StockSpider.py
3. 百度指数数据：FetchBindex

#### 训练集处理
1. 数据处理：CNN/DataHelper.py、CNN/News.py(新闻词典生成)
2. CNN训练数据集：CNN/OriginData.py
3. LSTM训练数据集：LSTM/OriginData.py
4. 集成学习模型训练数据集：Ensemble/Ensemble.py

#### 模型构建
1. CNN模型：CNN/CNNStockText.py、CNN/CNNStockNumber.py
2. LSTM模型：LSTM/LSTMStockOrigin.py
3. 集成学习模型：Ensemble/Ensemble.py

#### 交叉验证
1. 模型十折十次交叉验证：/main.py

#### 结果分析
1. 倾向词典分析：CNN/News.py
2. 模型预测结果分析：Spider/Analyse.py


#### 缺陷
1. 利用Ensemble类的makeEnsembleData函数生成集成模型的输入集时，因内存释放问题，在生成第一个输入集后速度会变慢（需生成100个输入集），暂处理方法为加入exit()并利用批处理（即每次生成一个后退出，利用批处理来执行一百遍）


#### 非代码相关资料
[实验数据](https://pan.baidu.com/s/1oLhhmJuDy7Tedp9i-puWpQ) (内含本人毕业论文、相关文献、数据库数据、实验过程数据、实验结果截图)

#### 版权声明
开源代码对学术研究完全开放，使用时请引用出处；请勿用作商业应用，后果自行负责。