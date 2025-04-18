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
from abc import ABC, abstractmethod

# 项目模块
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution, \
    SkewNormalDistribution, LogNormalDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.kernel.gaussian import \
    GaussianKernelMixDistribution
from data_utils.stochastic_utils.vdistributions.tools.clip import ClampedDistribution

from etrade.spot.trader import Station
from etrade.spot.market.recycle import BasicRecycle, PointwiseRecycle
from etrade.spot.forecast.market import DistributiveSeries, DistributiveMarket

# 外部模块
import numpy
from matplotlib import pyplot


# 代码块

class AbstractDistributionConstructor(ABC):
    @abstractmethod
    def random(self, *args, **kwargs):
        pass


class DistributionConstructor(AbstractDistributionConstructor):
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
        return dist_list


class OrdinaryGaussianKernelDistributionConstructor(AbstractDistributionConstructor):

    def __init__(self, mean_range=(0, 1), std_range=(1, 2), kernel_num_range=(1, 4)):
        self.mean_range = mean_range
        self.std_range = std_range
        self.kernel_num_range = kernel_num_range

    def random(self, num: int):
        dist_list = []
        for _ in range(num):
            kernel_num = numpy.random.randint(self.kernel_num_range[0], self.kernel_num_range[1])
            kernel_arg = []
            for k in range(kernel_num):
                kernel_arg.append([
                    numpy.random.uniform(self.mean_range[0], self.mean_range[1]),
                    numpy.random.uniform(self.std_range[0], self.std_range[1])
                ])
            dist_list.append(
                GaussianKernelMixDistribution(*kernel_arg)
            )
        return dist_list


class MarketConstructor:
    """市场构造器"""

    def __init__(
            self,
            aq_constructor: AbstractDistributionConstructor,
            dp_constructor: AbstractDistributionConstructor,
            rp_constructor: AbstractDistributionConstructor
    ):
        self.aq_constructor = aq_constructor
        self.dp_constructor = dp_constructor
        self.rp_constructor = rp_constructor

    def random(self, num: int):
        aq = DistributiveSeries(*self.aq_constructor.random(num))
        dp = DistributiveSeries(*self.dp_constructor.random(num))
        rp = DistributiveSeries(*self.rp_constructor.random(num))
        return DistributiveMarket(aq, dp, rp)

    def clamped_random(self, num, aq_range, dp_range, rp_range):
        aq = DistributiveSeries(*[ClampedDistribution(i, *aq_range) for i in self.aq_constructor.random(num)])
        dp = DistributiveSeries(*[ClampedDistribution(i, *dp_range) for i in self.dp_constructor.random(num)])
        rp = DistributiveSeries(*[ClampedDistribution(i, *rp_range) for i in self.rp_constructor.random(num)])
        return DistributiveMarket(aq, dp, rp)


if __name__ == "__main__":
    # kc = OrdinaryGaussianKernelDistributionConstructor(
    #     (20, 50), (1, 10), (2, 4)
    # )
    # r = kc.random(10)
    # for i in r:
    #     data = i.rvf(100)
    #     pyplot.hist(data)
    # pyplot.show()

    aq = OrdinaryGaussianKernelDistributionConstructor((20, 50), (1, 10), (2, 4))
    dp = OrdinaryGaussianKernelDistributionConstructor((0, 5), (1, 5), (2, 4))
    rp = OrdinaryGaussianKernelDistributionConstructor((0, 5), (1, 5), (2, 4))
    #
    mc = MarketConstructor(aq, dp, rp)
    mc.clamped_random(4, (0, 50), (0, numpy.inf), (0, numpy.inf)).plot()
    # mc.clamped_random(10, (0, 50), (0, numpy.inf), (0, numpy.inf))
    # #
    # rm = mc.random(4)
    # pm = mc.random(4)
    # # #
    # rs = rm.random_sample()
    # # # print(rs)
    # # #
    # print(pm.crps(rs[0], rs[1], rs[2]))
