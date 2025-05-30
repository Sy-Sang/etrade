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
        trade_yield_table = numpy.atleast_1d(trade_yield_table)
        submitted_quantity = numpy.atleast_2d(submitted_quantity)

        aq_sum = numpy.sum(actually_quantity_table, axis=0)

        if submitted_quantity.shape == actually_quantity_table.shape:
            sq_sum = numpy.sum(submitted_quantity, axis=0)
        else:
            sq_sum = numpy.sum(submitted_quantity)  # 总提交

        bias_mask = ((aq_sum / sq_sum) < self.bias_ratio) | ((aq_sum / sq_sum) > (2 - self.bias_ratio))
        penalty = trade_yield_table * self.penalty_coefficient
        adjusted_yield = trade_yield_table - penalty * bias_mask.astype(float)

        return adjusted_yield.reshape(-1)


class PointwiseRecycle(BasicRecycle):
    """逐点的回收机制"""

    # def penalty_q(self, aq_table, sq):
    #     """判断是否惩罚"""
    #     aq_table = numpy.atleast_2d(aq_table)
    #     sq = numpy.asarray(sq)
    #     if sq.shape == aq_table.shape:
    #         pass
    #     else:
    #         sq = numpy.expand_dims(sq, axis=1)
    #         sq = numpy.broadcast_to(sq, aq_table.shape)
    #     condition = (aq_table > (1 + self.bias_ratio) * sq) | (aq_table < self.bias_ratio * sq)
    #     # return numpy.any(condition, axis=0)
    #     return condition
    def penalty_q(self, aq_table, sq):
        """判断是否惩罚"""
        aq_table = numpy.atleast_2d(aq_table)
        sq = numpy.atleast_2d(sq)

        # 如果维度不匹配，进行广播处理
        if sq.shape != aq_table.shape:
            try:
                sq = numpy.broadcast_to(sq, aq_table.shape)
                # print("submitted_quantity 被广播为:", sq.shape)
            except ValueError:
                raise ValueError(
                    f"无法将 submitted_quantity 形状 {sq.shape} 广播为 actually_quantity_table 形状 {aq_table.shape}")

        # 判断是否超过阈值
        condition = (aq_table > (1 + self.bias_ratio) * sq) | (aq_table < self.bias_ratio * sq)
        return condition  # shape: same as aq_table (element-wise mask)

    def __call__(self, actually_quantity_table, submitted_quantity, trade_yield_table, *args, **kwargs):
        trade_yield_table = numpy.atleast_2d(trade_yield_table)

        penalty_mask = self.penalty_q(actually_quantity_table, submitted_quantity)
        # penalty = numpy.sum(trade_yield_table, axis=0) * self.penalty_coefficient
        # adjusted_yield = numpy.sum(trade_yield_table, axis=0) - penalty * penalty_mask.astype(float)
        # return adjusted_yield
        penalty = trade_yield_table * self.penalty_coefficient
        adjusted_yield = numpy.sum(trade_yield_table, axis=0) - numpy.sum(penalty * penalty_mask.astype(float), axis=0)
        return adjusted_yield


if __name__ == "__main__":
    br = BasicRecycle()
    print(br(50, 40, 100))

    pr = PointwiseRecycle()

    print(pr.penalty_q(numpy.arange(16).reshape(-1, 2), numpy.arange(0, 16, 2)))
