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

    def __init__(self, seq_len: int = 96, aggregation_len: int = 4):
        self.seq_len = seq_len
        self.aggregation_len = aggregation_len

    def to_seq(self, data) -> numpy.ndarray:
        data = numpy.asarray(data)
        if data.ndim == 0:
            data = numpy.pad(numpy.array([data]), (0, self.seq_len - 1))
        else:
            if len(data) > self.seq_len:
                data = data[:self.seq_len]
            else:
                data = numpy.pad(data, (0, self.seq_len - len(data)))
        return data

    def __call__(self, actually_quantity, submitted_quantity, trade_yield, *args, **kwargs):
        aq = self.to_seq(actually_quantity)
        sq = self.to_seq(submitted_quantity)
        ty = self.to_seq(trade_yield)


if __name__ == "__main__":
    pass
