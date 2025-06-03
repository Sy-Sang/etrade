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
from typing import Union, Self, Tuple
from collections import namedtuple

# 项目模块
from easy_utils.number_utils.calculus_utils import n_integrate
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution
from data_utils.stochastic_utils.vdistributions.tools.convert import generate_correlated_sample_matrix
from data_utils.stochastic_utils.vdistributions.tools.divergence import kl_divergence_continuous, crps, quantile_RMSE, \
    js_divergence_continuous

from etrade.spot.trader import Station
from etrade.spot.market.recycle import Recycle

# 外部模块
import numpy
from scipy.optimize import differential_evolution, minimize
from matplotlib import pyplot


# 代码块

class DistributiveSeries:
    """分布序列"""

    def __init__(self, *args: AbstractDistribution):
        self.distributions = copy.deepcopy(args)
        self.len = len(args)

    def rvf(self, num: int):
        """随机样本"""
        return numpy.stack([d.rvf(num) for d in self.distributions], axis=0)

    def mean(self, num: int):
        return numpy.mean(self.rvf(num), axis=1)

    def correlated_rvf(self, num: int, pearson: Tuple[float], sample=None):
        """带有相关性的随机样本"""
        sample = NormalDistribution(0, 1).rvf(num) if sample is None else numpy.asarray(sample)
        return generate_correlated_sample_matrix(
            sample,
            *[(self.distributions[i], pearson[i]) for i in range(self.len)]
        )


class DistributiveMarket:
    """市场内部变量"""

    def __init__(self, power_generation: DistributiveSeries, dayahead_price: DistributiveSeries,
                 realtime_price: DistributiveSeries):
        self.power_generation = copy.deepcopy(power_generation)
        self.dayahead_price = copy.deepcopy(dayahead_price)
        self.realtime_price = copy.deepcopy(realtime_price)
        self.shape = (3, self.power_generation.len)
        self.map = {
            0: self.power_generation,
            1: self.dayahead_price,
            2: self.realtime_price
        }
        self.market_len = self.power_generation.len

    def __repr__(self):
        return str({
            "power_generation": self.power_generation.distributions,
            "dayahead_price": self.dayahead_price.distributions,
            "realtime_price": self.realtime_price.distributions
        })

    def __getitem__(self, item) -> AbstractDistribution:
        if item[0] == 0:
            return self.power_generation.distributions[item[1]]
        elif item[0] == 1:
            return self.dayahead_price.distributions[item[1]]
        elif item[0] == 2:
            return self.realtime_price.distributions[item[1]]
        else:
            raise Exception(f"item mistake: {item}")

    def plot(self, curve_index=1, num=100):
        """显示pyplot"""
        counter = 1
        for i in range(3):
            for j in range(self.power_generation.len):
                pyplot.subplot(3, self.power_generation.len, counter)
                curve = self.map[i].distributions[j].curves(num)[curve_index]
                pyplot.plot(curve[:, 0], curve[:, 1])
                counter += 1
        pyplot.show()

    def plot2(self, curve_index=1, num=100):
        for i in range(self.market_len):
            pyplot.subplot(2, self.market_len, i + 1)
            curve = self[0, i].curves(num)[curve_index]
            pyplot.plot(curve[:, 0], curve[:, 1])

        for i in range(self.market_len):
            pyplot.subplot(2, self.market_len, i + self.market_len + 1)
            curve_dp = self[1, i].curves(num)[curve_index]
            curve_rp = self[2, i].curves(num)[curve_index]
            pyplot.plot(curve_dp[:, 0], curve_dp[:, 1])
            pyplot.plot(curve_rp[:, 0], curve_rp[:, 1])

        pyplot.show()

    def rvf(self, num: int, aq_range=(-numpy.inf, numpy.inf), dp_range=(-numpy.inf, numpy.inf),
            rp_range=(-numpy.inf, numpy.inf)):
        """随机样本"""
        return (numpy.clip(self.power_generation.rvf(num), *aq_range),
                numpy.clip(self.dayahead_price.rvf(num), *dp_range),
                numpy.clip(self.realtime_price.rvf(num), *rp_range))

    def observe(self):
        """观测数据"""
        return numpy.asarray(self.rvf(1)).reshape(3, -1)

    def random_sample(self, aq_range=(-numpy.inf, numpy.inf), dp_range=(-numpy.inf, numpy.inf),
                      rp_range=(-numpy.inf, numpy.inf)):
        aq, dp, rp = self.rvf(1, aq_range, dp_range, rp_range)
        aq = aq.reshape(self.shape[1])
        dp = dp.reshape(self.shape[1])
        rp = rp.reshape(self.shape[1])
        return numpy.stack([aq, dp, rp], axis=0)

    def mean(self, num: int):
        return self.power_generation.mean(num), self.dayahead_price.mean(num), self.realtime_price.mean(num)

    def correlated_rvf(self, num: int, pearson, samples=None):
        """带有相关性的随机样本, 有错误, 未clip"""
        samples = [
            [None] * self.power_generation.len,
            [None] * self.power_generation.len,
            [None] * self.power_generation.len,
        ] if samples is None else samples

        return (
            self.power_generation.correlated_rvf(num, pearson[0], samples[0]),
            self.dayahead_price.correlated_rvf(num, pearson[1], samples[1]),
            self.realtime_price.correlated_rvf(num, pearson[2], samples[2])
        )

    @classmethod
    def trade(cls, station: Station, aq, dp, rp, x):
        """交易"""
        x = numpy.asarray(x)
        if x.shape == aq.shape:
            pass
        else:
            x = numpy.expand_dims(x, axis=1)
            x = numpy.broadcast_to(x, aq.shape)
        return numpy.sum(station.trade(aq, x, dp, rp), axis=0)

    @classmethod
    def trade_with_recycle(cls, station: Station, recycle: Recycle, aq, dp, rp, x):
        """考虑回收机制的交易"""
        return recycle(aq, x, cls.trade(station, aq, dp, rp, x))


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
    from matplotlib import pyplot

    d = DistributiveSeries(NormalDistribution(0, 1), NormalDistribution(100, 10))
    pyplot.scatter(*d.correlated_rvf(10, [1, 1]))
    pyplot.show()
    # print(d.rvf(10))
