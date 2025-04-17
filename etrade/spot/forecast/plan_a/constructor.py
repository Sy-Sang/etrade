#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plan a的随机市场变量构造器"""

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
from typing import Union, Self, Type
from collections import namedtuple

# 项目模块
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution, \
    SkewNormalDistribution, LogNormalDistribution
from etrade.spot.trader import Station
from etrade.spot.market.recycle import BasicRecycle, PointwiseRecycle
from etrade.spot.forecast.market import DistributiveSeries, DistributiveMarket

# 外部模块
import numpy


# 代码块

class DistributionConstructor:
    """分布构造器"""

    def __init__(self, dist_type: Type[ParameterDistribution], param_range: list):
        self.dist_type = dist_type
        self.param_range = param_range

    def random(self, num: int):
        dist_list = []
        params = numpy.zeros((len(self.param_range), num))
        for i, pr in enumerate(self.param_range):
            params[i] = numpy.random.uniform(pr[0], pr[1], num)

        for i in range(num):
            dist_list.append(self.dist_type(*params[:, i]))
        return DistributiveSeries(*dist_list)


class MarketConstructor:
    """市场构造器"""

    def __init__(
            self,
            aq_constructor: DistributionConstructor,
            dp_constructor: DistributionConstructor,
            rp_constructor: DistributionConstructor
    ):
        self.aq_constructor = aq_constructor
        self.dp_constructor = dp_constructor
        self.rp_constructor = rp_constructor

    def random(self, num: int):
        aq = self.aq_constructor.random(num)
        dp = self.dp_constructor.random(num)
        rp = self.rp_constructor.random(num)
        return DistributiveMarket(aq, dp, rp)


if __name__ == "__main__":
    aq = DistributionConstructor(NormalDistribution, [(30, 50), (5, 10)])
    dp = DistributionConstructor(LogNormalDistribution, [(0, 1), (0.1, 0.2)])
    rp = DistributionConstructor(LogNormalDistribution, [(0, 1), (0.1, 0.2)])

    mc = MarketConstructor(aq, dp, rp)

    rm = mc.random(4)
    pm = mc.random(4)

    rs = rm.random_sample()
    print(rs)

    print(pm.crps(rs[0], rs[1], rs[2]))
