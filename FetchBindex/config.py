#coding=utf-8
import configparser as ConfigParser
import io
import os

def load_configs(column='config', config_ini_file='config.ini'):
    floder_path = os.path.dirname(os.path.abspath(__file__))
    _cp = ConfigParser.ConfigParser()
    fp = open(os.path.join(floder_path, config_ini_file), 'r', encoding='utf-8')
    content = fp.read()
    fp.close()
    fp = io.StringIO(content)
    _cp.readfp(fp)
    return _cp.items(column)


configs = dict(load_configs())

class IniConfig(object):

    def __getattribute__(self, *args, **kwargs):
        val = configs.get(args[0], '').strip()
        return val


ini_config = IniConfig()
