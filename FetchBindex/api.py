#coding=utf-8
import copy
import re
import requests

from .img_util import get_num


UserAgent = ('Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, '
             'like Gecko) Chrome/32.0.1700.76 Safari/537.36')

HUMAN_HEADERS = {
    'Accept': ('text/html,application/xhtml+xml,application/xml;q=0.9,'
               'image/webp,*/*;q=0.8'),
    'User-Agent': UserAgent,
    'Accept-Encoding': 'gzip,deflate,sdch'
}


class Api(object):
    def __init__(self, cookie):
        self.headers = copy.deepcopy(HUMAN_HEADERS)
        self.headers.update({'Cookie': cookie})

    def get_all_index_html(self, all_index_url):
        r = requests.get(all_index_url, headers=self.headers)
        return r.json()

    def get_index_show_html(self, index_show_url):
        r = requests.get(index_show_url, headers=self.headers)
        content = r.json()['data']['code'][0]
        img_url = re.findall('(?is)"(/Interface/IndexShow/img/[^"]*?)"', content)
        img_url = "http://index.baidu.com%s" % img_url[0]

        regex = ('(?is)<span class="imgval" style="width:(\d+)px;">'
                 '<div class="imgtxt" style="margin-left:-(\d+)px;">')
        result = re.findall(regex, content)
        skip_info = result if result else list()
        return img_url, skip_info

    def get_value_from_url(self, img_url, index_skip_info):
        r = requests.get(img_url, headers=self.headers)
        return get_num(r.content, index_skip_info)
