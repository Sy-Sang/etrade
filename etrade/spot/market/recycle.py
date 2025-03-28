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

    def __init__(self, seq_len: int = 96, aggregation_len: int = 4, bias_ratio: float = 0.5,
                 penalty_coefficient: float = 1.05):
        self.seq_len = seq_len
        self.aggregation_len = aggregation_len
        self.bias_ratio = bias_ratio
        self.penalty_coefficient = penalty_coefficient

    def to_seq(self, data) -> numpy.ndarray:
        """修整向量长度"""
        data = numpy.atleast_1d(numpy.asarray(data))
        data = data[:self.seq_len]
        return numpy.pad(data, (0, self.seq_len - data.size))

    def __call__(self, actually_quantity, submitted_quantity, trade_yield, *args, **kwargs):
        eps = numpy.finfo(float).eps
        aq = self.to_seq(actually_quantity)
        sq = self.to_seq(submitted_quantity)
        ty = self.to_seq(trade_yield)

        aq_grouped = aq.reshape(-1, self.aggregation_len)
        sq_grouped = sq.reshape(-1, self.aggregation_len)
        ty_grouped = numpy.sum(ty.reshape(-1, self.aggregation_len), axis=1)

        group_deviation = numpy.sum(sq_grouped + eps, axis=1) / numpy.sum(aq_grouped + eps, axis=1)
        recycle_index = numpy.where(
            (group_deviation <= (1 - self.bias_ratio)) | (group_deviation >= (1 + self.bias_ratio))
        )[0]
        return numpy.sum(ty_grouped) - numpy.sum(ty_grouped[recycle_index] * self.penalty_coefficient)


if __name__ == "__main__":
    br = BasicRecycle()
    print(br([50, 100], 1000, [100, 200]))
