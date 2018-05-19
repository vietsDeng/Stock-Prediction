#coding=utf-8
import os
import traceback
import sys

import xlwt
import chardet

import json

from .browser import BaiduBrowser
from .utils.log import logger
from .config import ini_config

class BindexSpider():

    index_type_dict = {
        'all': u'整体趋势', 'pc': u'PC趋势', 'wise': u'移动趋势'
    }

    FILE_NAME_ENCODING = 'utf-8'

    def save_cookie_to_file(self, cookie_json):
        with open(ini_config.cookie_file_path, 'w') as f:
            f.write(cookie_json)


    def load_cookie_from_file(self,):
        cookie_json = ''
        if os.path.exists(ini_config.cookie_file_path):
            with open(ini_config.cookie_file_path, 'r') as f:
                cookie_json = f.read()
        return cookie_json

    def craw(self, conn, cur):
        logger.info(u'请确保你填写的账号密码能够成功登陆百度')
        # 创建data目录
        result_folder = ini_config.out_file_path
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # 加载曾经保存的cookie文件,尽量避免重复登录
        cookie_json = self.load_cookie_from_file()
        baidu_browser = BaiduBrowser(cookie_json=cookie_json)
        # 将登陆成功后的cookie_json保存到文件
        self.save_cookie_to_file(baidu_browser.get_cookie_json())
        logger.info(u'登陆成功')
        
        cur.execute("select id, word, baidu_code from vocab where baidu_code is NULL and status = 1 order by id asc limit 90,200 ")
        word_data = cur.fetchall()

        cur.execute("select word, baidu_code from vocab where baidu_code = '0' and status = 1 order by baidu_code asc")
        city_data = cur.fetchall()

        area_list = []
        for d in city_data:
            area_list.append(str(d[1]))

        for d in word_data:
            try:
                keyword = d[1].strip()
                if not keyword:
                    continue
                baidu_data = self.parse_one_keyword(keyword, area_list, baidu_browser)
                
                for k in baidu_data:
                    #插入一条数据
                    cur.execute("insert into baidu_index (vocab_id, bindex, date) values('" + str(d[0])  + "','" + json.dumps(baidu_data[k]) + "','" + k + "')")                    
                conn.commit()
                del baidu_data

            except:
                print (traceback.format_exc())


    def parse_one_keyword(self, keyword, area_list, baidu_browser):
        if area_list is None:
            area_list = ini_config.area_list.split(',')
        area_list = [_.strip() for _ in area_list]
        type_list = ini_config.index_type_list.split(',')
        type_list = [_.strip() for _ in type_list]
        
        logger.info('%s start' % keyword)
        baidu_data = {}
        for area in area_list:
            for type_name in type_list:
                baidu_index_dict = baidu_browser.get_baidu_index(
                    keyword, type_name, area
                )

                for date in baidu_browser.date_list:
                    value = baidu_index_dict.get(date, 0)

                    baidu_data.setdefault(date,{})
                    baidu_data[date].setdefault(type_name,{})
                    baidu_data[date][type_name].setdefault(area, value)        

        return baidu_data

    def write_excel(self, excel_file, data_list):
        wb = xlwt.Workbook()
        ws = wb.add_sheet(u'工作表1')
        row = 0
        ws.write(row, 0, u'关键词')
        ws.write(row, 1, u'日期')
        ws.write(row, 2, u'类型')
        ws.write(row, 3, u'指数')
        row = 1
        for result in data_list:
            col = 0
            for item in result:
                ws.write(row, col, item)
                col += 1
            row += 1

        wb.save(excel_file)
