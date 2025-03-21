#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Sy,Sang"
__version__ = ""
__license__ = "GPLv3"
__maintainer__ = "Sy, Sang"
__email__ = "martin9le@163.com"
__status__ = "Development"
__credits__ = []
__date__ = ""
__copyright__ = ""

# 系统模块
import copy
import pickle
import json
from typing import Union, Self
from collections import namedtuple

# My github 模块


# 项目模块
from etrade.spot.resource.namedtuple import *
from etrade.spot.market.recycle import Recycle

# 外部模块
import numpy


# 代码块

class Station:
    """
    发电厂
    """

    def __init__(self, name: str, max_power: float):
        self.name = name
        self.max_power = max_power

    def trade(self, actually_quantity, submitted_quantity, dayhead_price, realtime_price) -> numpy.ndarray:
        """计算电力市场交易收益（支持批量计算）"""
        actually_quantity = numpy.asarray(actually_quantity)
        submitted_quantity = numpy.asarray(submitted_quantity)
        dayhead_price = numpy.asarray(dayhead_price)
        realtime_price = numpy.asarray(realtime_price)

        aq = numpy.clip(actually_quantity, 0, self.max_power)
        sq = numpy.clip(submitted_quantity, 0, self.max_power)
        rq = aq - sq

        return sq * dayhead_price + rq * realtime_price


if __name__ == "__main__":
    s = Station("s", 50)
    aq = numpy.random.uniform(0, 100, 96)
    sq = numpy.random.uniform(0, 100, 96)

    print(
        numpy.column_stack((
            aq,
            sq,
            s.trade(30, sq, 1, 6)
        ))
    )
