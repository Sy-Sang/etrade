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
from abc import ABC, abstractmethod

# 项目模块

# 外部模块
import numpy


# 代码块

class Recycle(ABC):
    """超额回收机制(抽象类)"""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BasicRecycle(Recycle):

    def __init__(self, bias_ratio: float = 0.5,
                 penalty_coefficient: float = 1.05):
        self.bias_ratio = bias_ratio
        self.penalty_coefficient = penalty_coefficient

    def __call__(self, actually_quantity_table, submitted_quantity, trade_yield_table, *args, **kwargs):
        actually_quantity_table = numpy.atleast_2d(actually_quantity_table)
        trade_yield_table = numpy.atleast_2d(trade_yield_table)

        aq_sum = numpy.sum(actually_quantity_table, axis=0)  # 每列求和
        sq_sum = numpy.sum(submitted_quantity)

        bias_mask = (aq_sum > (1 + self.bias_ratio) * sq_sum) | (aq_sum < self.bias_ratio * sq_sum)
        penalty = numpy.sum(trade_yield_table, axis=0) * self.penalty_coefficient

        adjusted_yield = numpy.sum(trade_yield_table, axis=0) - penalty * bias_mask.astype(float)
        return adjusted_yield


if __name__ == "__main__":
    br = BasicRecycle()
    print(br(50, 40, 100))
