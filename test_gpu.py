# -*- coding:utf-8 -*-
"""
@author: xinquan
@file: test_gpu.py
@time: 2021/7/21 10:12 下午
@desc: 
"""
import os
import warnings

warnings.filterwarnings('ignore')  # 忽略一些警告,可以删除
root_path = os.path.split(os.path.realpath(__file__))[0]  # 获取该脚本的地址,有效避免Linux和Windows文件路径格式不一致等问题,可以删除
