#coding=utf-8
import pymysql

class VMysql:
    @classmethod
    def get_mysql(cls):
        conn = pymysql.connect(
            host='localhost',
            port = 3306,
            user='root',
            passwd='xxx',
            db ='prediction',
            charset='utf8',
        )

        cur = conn.cursor()
        cur.execute("SET NAMES utf8")
        cur.execute("set group_concat_max_len = 10240000")

        return conn,cur