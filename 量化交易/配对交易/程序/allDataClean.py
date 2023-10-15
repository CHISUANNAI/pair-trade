#-*- coding : utf-8 -*-
# coding: utf-8

import shutil

path = '..\\result'
try:
    shutil.rmtree(path,ignore_errors=True)
    print("删除成功")
except:
    print("删除失败")