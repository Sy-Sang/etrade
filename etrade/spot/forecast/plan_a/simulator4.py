#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""模拟器v3.0"""

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
from easy_utils.number_utils.number_utils import EasyFloat
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.kernel.gaussian import \
    GaussianKernelWeightedMixDistribution
from etrade.spot.forecast.market import DistributiveSeries, DistributiveMarket
from etrade.spot.forecast.plan_a.constructor import *
from etrade.spot.forecast.yieldindex import zero_quantile

from etrade.spot.forecast.plan_a.simulator import MarketSimulator
from etrade.spot.market.recycle import BasicRecycle, PointwiseRecycle

# 外部模块
import numpy


# 代码块


class PredictBasedMarketSimulator:
    """
    基于预测市场的模拟器
    先构造一个预测市场, 再通过加入预测噪音, 形成多个真实市场场景
    模拟针对预测市场的解, 在什么样的预测噪音环境下是依然奏效的
    """

    def __init__(
            self,
            aq_constructor: OrdinaryGaussianKernelDistributionConstructor,
            dp_constructor: OrdinaryGaussianKernelDistributionConstructor,
            rp_constructor: OrdinaryGaussianKernelDistributionConstructor,
            aq_range=(-numpy.inf, numpy.inf),
            dp_range=(-numpy.inf, numpy.inf),
            rp_range=(-numpy.inf, numpy.inf),
            predict_weight=numpy.ones((3, 4)),
            noise_weight_range=((0.01, 0.05), (0.01, 0.05), (0.01, 0.05)),
            market_len=4,
    ):
        self.aq_constructor = copy.deepcopy(aq_constructor)
        self.dp_constructor = copy.deepcopy(dp_constructor)
        self.rp_constructor = copy.deepcopy(rp_constructor)
        self.mc = MarketConstructor(self.aq_constructor, self.dp_constructor, self.rp_constructor)
        self.predict_weight = predict_weight
        self.predict_market = self.mc.random(market_len)
        self.noise_weight_range = noise_weight_range

        self.aq_range = aq_range
        self.dp_range = dp_range
        self.rp_range = rp_range
        self.market_len = market_len

    def predicted_random(self, rounds=1000):
        power_generation, dayahead_price, realtime_price = self.predict_market.rvf(
            rounds, self.aq_range, self.dp_range, self.rp_range
        )
        return power_generation, dayahead_price, realtime_price

    def predicted_optimize(self, station: Station, recycle: BasicRecycle, power_generation, dayahead_price,
                           realtime_price):
        return DistributiveMarket.submitted_quantity_optimizer(
            station, recycle, power_generation, dayahead_price, realtime_price
        ).x

    def random_sample_trade(
            self, x, station: Station, recycle: BasicRecycle, power_generation, dayahead_price, realtime_price
    ):
        sq = numpy.asarray(x)
        return recycle(
            power_generation,
            sq,
            DistributiveMarket.trade(station, power_generation, dayahead_price, realtime_price, sq)
        )

    # def new_divergenced_market(self, kl_divergence=(1, 1, 1)):
    #     def divergenced_kernel(dist:GaussianKernelMixDistribution, kl_divergence_value:float):
    #         kernel_matrix = dist.kernel_data().reshape(-1)


if __name__ == "__main__":
    from easy_datetime.timestamp import TimeStamp

    t0 = TimeStamp.now()

    market_len = 4
    init_kwargs = {
        "aq_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 50), (1, 10), (1, 4)),
        "dp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 20), (1, 10), (1, 4)),
        "rp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 20), (1, 10), (1, 4)),
        "aq_range": (0, 50),
        "dp_range": (0, 1e+6),
        "rp_range": (0, 1e+6),
        "predict_weight": numpy.full((3, market_len), 1),
        "noise_weight_range": ((0.01, 0.05), (0.01, 0.05), (0.01, 0.05)),
        "market_len": market_len
    }

    station = Station("station", 50)
    br = BasicRecycle(0.5, 1.05)
    simulator = PredictBasedMarketSimulator(**init_kwargs)

    aq, dp, rp = simulator.predicted_random(1000)

    print(numpy.mean(aq, axis=1).tolist())
    x = simulator.predicted_optimize(station, br, aq, dp, rp)
    print(x.tolist())
    aq, dp, rp = simulator.predicted_random(1000)
    # trade_aq = numpy.mean(aq, axis=1)
    trade_aq = aq

    ppf = numpy.sort(
        simulator.random_sample_trade(x, station, br, aq, dp, rp) - simulator.random_sample_trade(trade_aq, station, br,
                                                                                                  aq, dp, rp)
    )
    pyplot.plot(simulator.random_sample_trade(x, station, br, aq, dp, rp))
    pyplot.plot(simulator.random_sample_trade(trade_aq, station, br, aq, dp, rp))
    # pyplot.plot([0] * 1000)
    print(numpy.mean(simulator.random_sample_trade(x, station, br, aq, dp, rp)))
    print(numpy.mean(simulator.random_sample_trade(trade_aq, station, br, aq, dp, rp)))
    pyplot.show()

    simulator.predict_market.plot()

    print(TimeStamp.now() - t0)
