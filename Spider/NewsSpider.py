#coding=utf-8
import pymysql
import requests
import re
import json
import time
import datetime

class NewsSpider():
    def craw(self, conn, cur):
        insertContentSql="INSERT INTO news (title, content, time, news_from)VALUES(%s,%s,%s,%s)"

        s = requests.session()
        s.keep_alive = False

        reArticle = re.compile('<td><a href="(.+?)" target="">\s*<span class="ClipItemTitle">.+?<\/span>')
        reFrom = re.compile('<td align="center" class="pubName">\s*(.+?)&nbsp;\s*<\/td>')
        reTitle = re.compile('<td colspan="3" align="left" class="headline">\s*(.+?)\s*<\/td>')
        reTime = re.compile('<td.*?class="info".*?>.*?<br>\s*(\d+-\d+-\d+)\s*<\/td>')
        reContent = re.compile('<td.*?class="content".*?>\s*(.+?)\s*<\/td>')
        dr = re.compile(r'<[^>]+>',re.S)
        drr = re.compile(r'\u3000',re.S)

        cookieFile = open('cookie.txt', 'r')
        cookie = cookieFile.read()
        headers = { 'Cookie': cookie}

        urlTypes = ['65029']

        #2012 - 2017 40W+ 会断线
        #20100101 - 20100209
        #2012 -2014
        #2014-01-08
        #12 | 14
        for t in urlTypes:
            url = 'http://wisenews.wisers.net.cn/wisenews/content.do?wp_dispatch=menu-content&menu-id=/commons/CFT-CN/DA000-DA200-DA210-/DA000-DA200-DA210-' + t + '-&srp_save&cp&cp_s=%d&cp_e=%d'
            for i in range(0, 10000000, 50):
                html = requests.get(url % (i,i+50), headers = headers).text
                hrefs = reArticle.findall(html)

                print(str(i) + ' articles')        
                
                for h in hrefs:
                    try:
                        html = requests.get('http://wisenews.wisers.net.cn' + h, headers = headers).text

                        newsFrom = reFrom.findall(html)            
                        title = reTitle.findall(html)
                        time = reTime.findall(html)
                        content = reContent.findall(html)

                        newsFrom = newsFrom[0] if len(newsFrom) > 0 else ''
                        title = drr.sub('',dr.sub('',title[0])) if len(title) > 0 else ''
                        content = drr.sub('',dr.sub('',content[0])) if len(content) > 0 else ''
                        time = time[0] if len(time) > 0 else ''

                        if time == '' or content == '' or title == '':
                            continue

                        index = content.rfind('原文连接')
                        if index != -1:
                            content = content[0:index]

                        d = [title, content, time, newsFrom]
                        print(d)
                        # cur.execute(insertContentSql, d)
                        # conn.commit()

                    except:
                        pass

                if len(hrefs) < 48:
                    break
