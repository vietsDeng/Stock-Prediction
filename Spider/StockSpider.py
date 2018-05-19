#coding=utf-8
import requests
import re
import json
import time

#zs_000001  上证指数
#zs_399001  深证成指
class StockSpider:
    def crawStockHistoryBySoHu(self, code, ctype = 'cn', start = '20100101', end = '20171201'):
        url = 'http://q.stock.sohu.com/hisHq?code=' + ctype + '_' + code + '&start=' + start + '&end=' + end + '&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp'    
        html = requests.get(url).text    
        html = html[html.index('{'): html.rindex('}') + 1]
        data = []
        if html != '{}':    	
            data = json.loads(html)
            data = data['hq']        
        return data

    def craw(self, conn, cur, fields):
        #fields = [['000001', 'zs', '上证指数'], ['399001', 'zs', '深证成指']]        
        insertStockSql = 'INSERT INTO stock (`name`, `code`) VALUES (%s, %s)'
        insertContentSql = 'INSERT INTO history (`stock_id`, `date`, `opening`, `closing`, `difference`, `percentage_difference`, `lowest`, `highest`, `volume`, `amount`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
        for f in fields:
            cur.execute(insertStockSql, [f[2], f[0]])
            stockId = conn.insert_id()
            data = self.crawStockHistoryBySoHu(code = f[0], ctype = f[1])
            for d in data:
                # print(d)
                cur.execute(insertContentSql, [stockId, d[0], d[1], d[2], d[3], d[4][0:-1], d[5], d[6], d[7], d[8]])
                conn.commit()
