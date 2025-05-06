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
from typing import Union, Self, Type, Iterable
from collections import namedtuple
from abc import ABC, abstractmethod

# 项目模块
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution, \
    SkewNormalDistribution, LogNormalDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.kernel.gaussian import \
    GaussianKernelMixDistribution, GaussianKernelWeightedMixDistribution
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.kernel2 import KernelMixDistribution
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.histogram import HistogramDistribution
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
    """平凡的高斯核分布构造器"""

    def __init__(self, mean_range=(0, 1), std_range=(1, 2), kernel_num_range=(1, 4)):
        self.mean_range = mean_range
        self.std_range = std_range
        self.kernel_num_range = kernel_num_range

    def random(self, num: int):
        dist_list = []
        for _ in range(num):
            kernel_num = numpy.random.randint(*self.kernel_num_range)
            kernel_arg = []
            for k in range(kernel_num):
                kernel_arg.append([
                    numpy.random.uniform(*self.mean_range),
                    numpy.random.uniform(*self.std_range)
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


def market_hybridization(market_a: DistributiveMarket, market_b: DistributiveMarket, num_a, num_b, kernel_num=None):
    def kernel_or_his(data):
        try:
            return KernelMixDistribution(data, kernel_num=kernel_num)
        except Exception as e:
            # print(f"[Fallback] KernelMix failed: {e} — switching to HistogramDistribution")
            print(f"[Fallback] KernelMix failed: — switching to HistogramDistribution")
            return HistogramDistribution(data)

    aq_a, dp_a, rp_a = market_a.rvf(num_a)
    aq_b, dp_b, rp_b = market_b.rvf(num_b)
    aq = numpy.column_stack((aq_a, aq_b))
    dp = numpy.column_stack((dp_a, dp_b))
    rp = numpy.column_stack((rp_a, rp_b))
    aq_list = []
    dp_list = []
    rp_list = []
    for i in range(len(aq)):
        aq_list.append(kernel_or_his(aq[i]))
        dp_list.append(kernel_or_his(dp[i]))
        rp_list.append(kernel_or_his(rp[i]))
    if num_b == 0:
        return copy.deepcopy(market_a)
    else:
        aq_series = DistributiveSeries(*aq_list)
        dp_series = DistributiveSeries(*dp_list)
        rp_series = DistributiveSeries(*rp_list)
    return DistributiveMarket(aq_series, dp_series, rp_series)


def market_hybridization_by_weight(
        market_a: DistributiveMarket,
        market_b: DistributiveMarket,
        num_a: numpy.ndarray,
        num_b: numpy.ndarray
):
    def weight_hybridization(d_a: GaussianKernelMixDistribution, d_b: GaussianKernelMixDistribution, w_r_a, w_r_b):
        k_a = d_a.kernel_data(None)
        w_a = numpy.full(k_a.shape[0], w_r_a)
        k_b = d_b.kernel_data(None)
        w_b = numpy.full(k_b.shape[0], w_r_b)
        arg_a = numpy.column_stack((k_a, w_a))
        arg_b = numpy.column_stack((k_b, w_b))
        args = numpy.concatenate((arg_a, arg_b))
        return GaussianKernelWeightedMixDistribution(*args)

    def series_hybridization(s_a: DistributiveSeries, s_b: DistributiveSeries, w_r_a, w_r_b):
        d = []
        for i in range(s_a.len):
            d.append(weight_hybridization(
                s_a.distributions[i], s_b.distributions[i], w_r_a[i], w_r_b[i]
            ))
        return DistributiveSeries(*d)

    pg = series_hybridization(market_a.power_generation, market_b.power_generation, num_a[0], num_b[0])
    dp = series_hybridization(market_a.dayahead_price, market_b.dayahead_price, num_a[1], num_b[1])
    rp = series_hybridization(market_a.realtime_price, market_b.realtime_price, num_a[2], num_b[2])
    return DistributiveMarket(pg, dp, rp)


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.kernel2 import silverman_bandwidth

    for _ in range(100):
        dist = OrdinaryGaussianKernelDistributionConstructor((0, 50), (0.1, 50), (2, 8)).random(4)
        for d in dist:
            try:
                kd = KernelMixDistribution(d.rvf(1000))
            except:
                print(d.kernel_data(1))
