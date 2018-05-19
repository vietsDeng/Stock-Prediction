#coding=utf-8
import traceback
import time
import urllib.parse
import json
from datetime import datetime, timedelta

from selenium import webdriver
from threadpool import ThreadPool, makeRequests

from .config import ini_config
from .api import Api
from .utils.log import logger


class BaiduBrowser():
    def __init__(self, cookie_json='', check_login=True):
        if not ini_config.browser_driver:
            browser_driver_name = 'Firefox'
        else:
            browser_driver_name = ini_config.browser_driver
        browser_driver_class = getattr(webdriver, browser_driver_name)

        if ini_config.browser_driver == 'Chrome':
            self.browser = browser_driver_class(executable_path=ini_config.executable_path)
        else:
            self.browser = browser_driver_class()
        # 设置超时时间
        self.browser.set_page_load_timeout(50)
        # 设置脚本运行超时时间
        self.browser.set_script_timeout(10)
        # 百度用户名
        self.user_name = ini_config.user_name
        # 百度密码
        self.password = ini_config.password
        self.cookie_json = cookie_json
        self.api = None
        self.cookie_dict_list = []
        self.date_list = []

        self.init_api(check_login=check_login)
        self.init_date_list()

    def __del__(self):
        self.close()

    def is_login(self):
        # 如果初始化BaiduBrowser时传递了cookie信息，则检测一下是否登录状态
        self.login_with_cookie(self.cookie_json)
        # 访问待检测的页面
        self.browser.get(ini_config.user_center_url)
        html = self.browser.page_source
        # 检测是否有登录成功标记
        return ini_config.login_sign in html

    def init_api(self, check_login=True):
        # 判断是否需要登录
        need_login = False
        if not self.cookie_json:
            logger.info(u'因无历史cookie，本次执行需要登录百度')
            need_login = True
        elif check_login and not self.is_login():
            logger.info(u'加载历史cookie登录失败，本次执行需要登录百度')
            need_login = True
        else:
            logger.info(u'本次执行无需登录百度')
        # 执行浏览器自动填表登录，登录后获取cookie
        if need_login:
            self.login(self.user_name, self.password)
            self.cookie_json = self.get_cookie_json()
        cookie_str = self.get_cookie_str(self.cookie_json)
        # 获取到cookie后传给api
        self.api = Api(cookie_str)

    def get_date_info(self, start_date, end_date):
        # 如果start_date和end_date中带有“-”，则替换掉
        if start_date.find('-') != -1 and end_date.find('-') != -1:
            start_date = start_date.replace('-', '')
            end_date = end_date.replace('-', '')
        # start_date和end_date转换成datetime对象
        start_date = datetime.strptime(start_date, '%Y%m%d')
        end_date = datetime.strptime(end_date, '%Y%m%d')

        # 循环start_date和end_date的差值，获取区间内所有的日期
        date_list = []
        temp_date = start_date
        while temp_date <= end_date:
            date_list.append(temp_date.strftime("%Y-%m-%d"))
            temp_date += timedelta(days=1)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        return start_date, end_date, date_list

    def get_one_day_index(self, date, url, keyword, type_name, area):
        try_num = 0
        try_max_num = 5
        value = ''
        while try_num < try_max_num:            
            try_num += 1
            # 获取图片的下载地址以及图片的切割信息
            img_url, val_info = self.api.get_index_show_html(url)
            # 下载img图片，然后根据css切割图片的信息去切割图片，组成新的图片，
            # 将新图片跟已经做好的图片识别库对应识别
            value = self.api.get_value_from_url(img_url, val_info)
            break            
        logger.info(
            'keyword:%s, type_name:%s, area:%s, date:%s, value:%s' % (
                keyword, type_name, area, date, value
            )
        )
        return value.replace(',', '')

    def get_baidu_index_by_date_range(self, keyword, start_date, end_date,
                                      type_name, area):
        # 根据区间获取关键词的索引值
        url = ini_config.time_range_trend_url.format(
            start_date=start_date, end_date=end_date,
            word=urllib.parse.quote(keyword.encode('gbk')),
            area=area
        )
        self.browser.get(url)
        if ini_config.browser_sleep:
            time.sleep(float(ini_config.browser_sleep))
        if u'未被收录' in self.browser.page_source:
            return {}
        # 执行js获取后面所需的res和res2的值
        res = self.browser.execute_script('return PPval.ppt;')
        res2 = self.browser.execute_script('return PPval.res2;')

        # 获取指定区间的日期列表,方便下面循环用
        start_date, end_date, date_list = self.get_date_info(
            start_date, end_date
        )

        # 拼接api的url
        url = ini_config.all_index_url.format(
            res=res, res2=res2, start_date=start_date, end_date=end_date
        )
        # 获取api的结果信息，这里面保存了后面日期节点的一些加密值
        all_index_info = self.api.get_all_index_html(url)        
        indexes_enc = all_index_info['data'][type_name][0]['userIndexes_enc']
        enc_list = indexes_enc.split(',')

        pool = ThreadPool(int(ini_config.num_of_threads))
        # wm = WorkManager(int(ini_config.num_of_threads))

        # 遍历这些enc值，这些值拼接出api的url(这个页面返回 图片信息以及css规定的切图信息)
        list_of_args = []
        for index, _ in enumerate(enc_list):
            url = ini_config.index_show_url.format(
                res=res, res2=res2, enc_index=_, t=int(time.time()) * 1000
            )
            # 根据enc在列表中的位置，获取它的日期
            date = date_list[index]
            # 将任务添加到多线程下载模型中
            item = (None, dict(date=date, url=url, keyword=keyword, type_name=type_name, area=area))
            list_of_args.append(item)

        baidu_index_dict = {}

        def callback(*args, **kwargs):
            req, val = args[0], args[1]
            baidu_index_dict[req.kwds['date']] = val

        req_list = makeRequests(self.get_one_day_index, list_of_args, callback)
        [pool.putRequest(req) for req in req_list]
        pool.wait()

        return baidu_index_dict

    def _get_index_period(self, keyword, area):
        # 拼接一周趋势的url
        url = ini_config.one_week_trend_url.format(
            area=area, word=urllib.parse.quote(keyword)
        )
        self.browser.get(url)
        # 获取下方api要用到的res和res2的值
        res = self.browser.execute_script('return PPval.ppt;')
        res2 = self.browser.execute_script('return PPval.res2;')
        start_date, end_date = self.browser.execute_script(
            'return BID.getParams.time()[0];'
        ).split('|')
        start_date, end_date, date_list = self.get_date_info(
            start_date, end_date
        )
        url = ini_config.all_index_url.format(
            res=res, res2=res2, start_date=start_date, end_date=end_date
        )
        all_index_info = self.api.get_all_index_html(url)
        start_date, end_date = all_index_info['data']['all'][0][
            'period'].split('|')
        # 重置start_date, end_date，以api返回的为准
        start_date, end_date, date_list = self.get_date_info(
            start_date, end_date
        )
        logger.info(
            'all_start_date:%s, all_end_date:%s' % (start_date, end_date)
        )
        return date_list

    def init_date_list(self):
        if ini_config.start_date and ini_config.end_date:
            _, _, date_list = self.get_date_info(
                start_date=ini_config.start_date, end_date=ini_config.end_date
            )
        else:
            # 配置文件不配置start_date和end_date，就会按照索引的最大区间来,
            # 目前每个关键词的最大区间是一样的，都是从2011年1月1日开始的
            date_list = self._get_index_period(keyword=u'test', area=0)
        self.date_list = date_list

    def get_baidu_index(self, keyword, type_name, area):
        baidu_index_dict = dict()
        start = 0
        skip = 190
        end = len(self.date_list)
        while start < end:
            try:
                start_date = self.date_list[start]
                if start + skip >= end - 1:
                    end_date = self.date_list[-1]
                else:
                    end_date = self.date_list[start + skip]
                result = self.get_baidu_index_by_date_range(
                    keyword, start_date, end_date, type_name, area
                )
                baidu_index_dict.update(result)
                start += skip + 1
            except:
                import traceback
                print (traceback.format_exc())
        return baidu_index_dict

    def login(self, user_name, password):
        login_url = ini_config.login_url
        # 访问登陆页
        self.browser.get(login_url)
        time.sleep(2)

        # 自动填写表单并提交，如果出现验证码需要手动填写
        while 1:
            try:
                user_name_obj = self.browser.find_element_by_id(
                    'TANGRAM__PSP_3__userName'
                )
                break
            except:
                logger.error(traceback.format_exc())
                time.sleep(1)
        user_name_obj.send_keys(user_name)
        ps_obj = self.browser.find_element_by_id('TANGRAM__PSP_3__password')
        ps_obj.send_keys(password)
        sub_obj = self.browser.find_element_by_id('TANGRAM__PSP_3__submit')
        sub_obj.click()

        # 如果页面的url没有改变，则继续等待
        while self.browser.current_url == login_url:
            time.sleep(1)

    def close(self):
        if getattr(self, 'browser'):
            if self.browser:
                try:
                    self.browser.quit()
                except:
                    pass

    def get_cookie_json(self):
        return json.dumps(self.browser.get_cookies())

    def get_cookie_str(self, cookie_json=''):
        if cookie_json:
            cookies = json.loads(cookie_json)
        else:
            cookies = self.browser.get_cookies()
        return '; '.join(['%s=%s' % (item['name'], item['value'])
                          for item in cookies])

    def login_with_cookie(self, cookie_json):
        self.browser.get('https://www.baidu.com/')
        for item in json.loads(cookie_json):
            try:
                self.browser.add_cookie(item)
            except:
                continue
