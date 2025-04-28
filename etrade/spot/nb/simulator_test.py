#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""nb6"""

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
import multiprocessing
import time

import os

from functools import partial

# 项目模块
from etrade.spot.forecast.market import DistributiveSeries, DistributiveMarket
from etrade.spot.forecast.plan_a.constructor import *
from etrade.spot.forecast.yieldindex import zero_quantile

# 外部模块
import numpy


# 代码块

def matched_gaussian_kernel_distribution_builder(kernel_data):
    """同水平高斯核混合分布构造器"""
    mu_data = kernel_data[:, 0]
    std_data = kernel_data[:, 1]
    l = len(kernel_data)
    mu_dist = NormalDistribution(numpy.mean(mu_data), numpy.std(mu_data, ddof=1))
    std_dist = NormalDistribution(numpy.mean(std_data), numpy.std(std_data, ddof=1))
    return GaussianKernelMixDistribution(*numpy.column_stack((mu_dist.rvf(l), std_dist.rvf(l))))


class MarketSimulator:
    def __init__(
            self,
            aq_constructor: OrdinaryGaussianKernelDistributionConstructor,
            dp_constructor: OrdinaryGaussianKernelDistributionConstructor,
            rp_constructor: OrdinaryGaussianKernelDistributionConstructor,
            aq_range=(-numpy.inf, numpy.inf),
            dp_range=(-numpy.inf, numpy.inf),
            rp_range=(-numpy.inf, numpy.inf),
            n0=1000, n1=1000, len=4
    ):
        self.mc = MarketConstructor(aq_constructor, dp_constructor, rp_constructor)
        self.n0 = n0
        self.n1 = n1
        self.real_market = self.mc.random(4)
        self.noice_market = self.mc.random(4)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.n0, self.n1)
        self.aq_range = aq_range
        self.dp_range = dp_range
        self.rp_range = rp_range
        self.len = len

    def refresh(self):
        self.real_market = self.mc.random(self.len)
        self.noice_market = self.mc.random(self.len)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.n0, self.n1)

    def replicate_noice_bandwidth_refresh(self):
        """保留噪音bandwidth的refresh"""
        aq_dist_list: list[GaussianKernelMixDistribution] = list(self.noice_market.power_generation.distributions)
        dp_dist_list: list[GaussianKernelMixDistribution] = list(self.noice_market.dayahead_price.distributions)
        rp_dist_list: list[GaussianKernelMixDistribution] = list(self.noice_market.realtime_price.distributions)
        for i in range(self.len):
            aq_dist_list[i] = matched_gaussian_kernel_distribution_builder(aq_dist_list[i].kernel_data())
            dp_dist_list[i] = matched_gaussian_kernel_distribution_builder(dp_dist_list[i].kernel_data())
            rp_dist_list[i] = matched_gaussian_kernel_distribution_builder(rp_dist_list[i].kernel_data())
        aq = DistributiveSeries(*aq_dist_list)
        dp = DistributiveSeries(*dp_dist_list)
        rp = DistributiveSeries(*rp_dist_list)
        self.noice_market = DistributiveMarket(aq, dp, rp)
        self.real_market = self.mc.random(self.len)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.n0, self.n1)

    def observe(self):
        """真实市场观察"""
        return self.real_market.random_sample(self.aq_range, self.dp_range, self.rp_range)

    def observed_crps(self):
        data = self.observe()
        crps = self.predicted_market.faster_crps(*data)
        return crps

    def random(self, rounds=1000):
        power_generation, dayahead_price, realtime_price = self.predicted_market.rvf(
            rounds, self.aq_range, self.dp_range, self.rp_range
        )
        return power_generation, dayahead_price, realtime_price

    def optimize(self, station: Station, recycle: BasicRecycle, rounds=1000):
        return self.predicted_market.submitted_quantity_optimizer(station, recycle, *self.random(rounds)).x

    def quantile(self, station: Station, recycle: BasicRecycle, rounds=1000):
        x = self.optimize(station, recycle, rounds)
        power_generation, dayahead_price, realtime_price = self.real_market.rvf(
            rounds, self.aq_range, self.dp_range, self.rp_range
        )
        return self.real_market.trade_with_recycle(station, recycle, power_generation, dayahead_price, realtime_price,
                                                   x), self.real_market.trade_with_recycle(
            station, recycle, power_generation, dayahead_price, realtime_price, self.real_market.mean(rounds)[0]
        )

    def zero_quantile(self, station: Station, recycle: BasicRecycle, rounds=1000):
        x = self.optimize(station, recycle, rounds)
        power_generation, dayahead_price, realtime_price = self.real_market.rvf(
            rounds, self.aq_range, self.dp_range, self.rp_range
        )
        return zero_quantile(
            self.real_market.trade_with_recycle(station, recycle, power_generation, dayahead_price, realtime_price, x),
            self.real_market.trade_with_recycle(
                station, recycle, power_generation, dayahead_price, realtime_price, self.real_market.mean(rounds)[0]
            )
        )


if __name__ == "__main__":
    init_kwargs = {
        "aq_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 50), (0.1, 10), (1, 8)),
        "dp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 10), (0.1, 10), (1, 8)),
        "rp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 10), (0.1, 10), (1, 8)),
        "aq_range": (0, 50),
        "dp_range": (0, 1e+6),
        "rp_range": (0, 1e+6),
        "n0": 1000,
        "n1": 10,
        "len": 4
    }
    s = Station("station", 50)
    br = PointwiseRecycle(0.5, 1.05)

    ms = MarketSimulator(**init_kwargs)

    a, b = ms.quantile(s, br)
    pyplot.plot(numpy.sort(a - b))
    # pyplot.plot(numpy.sort(b))
    pyplot.show()

    print(ms.predicted_market.price_kl_divergence())
    print(zero_quantile(a, b))
