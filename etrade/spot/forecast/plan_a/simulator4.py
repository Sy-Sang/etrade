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
            # noise_weight=numpy.ones((3, 4)),
            noise_weight_range=((0.01, 0.05), (0.01, 0.05), (0.01, 0.05)),
            market_len=4,
            p_head=3
    ):
        self.aq_constructor = copy.deepcopy(aq_constructor)
        self.dp_constructor = copy.deepcopy(dp_constructor)
        self.rp_constructor = copy.deepcopy(rp_constructor)
        self.mc = MarketConstructor(self.aq_constructor, self.dp_constructor, self.rp_constructor)
        self.predict_weight = predict_weight
        # self.noise_weight = noise_weight
        self.predict_market = self.mc.random(market_len)
        # self.noise_market_list = []
        # self.predicted_market_list = []
        self.noise_weight_range = noise_weight_range

        self.aq_range = aq_range
        self.dp_range = dp_range
        self.rp_range = rp_range
        self.market_len = market_len
        self.p_head = p_head

        # for _ in range(self.p_head):
        #     noise_market = self.mc.random(market_len)
        #     predicted_market = market_hybridization_by_weight(
        #         self.predict_market, noise_market, self.predict_weight, self.noise_weight
        #     )
        #     self.noise_market_list.append(noise_market)
        #     self.predicted_market_list.append(predicted_market)

    # def refresh(self):
    #     self.predict_market = self.mc.random(self.market_len)
    #     self.noise_market_list = []
    #     self.predicted_market_list = []
    #     for _ in range(self.p_head):
    #         noise_market = self.mc.random(self.market_len)
    #         predicted_market = market_hybridization_by_weight(
    #             self.predict_market, noise_market, self.predict_weight, self.noise_weight
    #         )
    #         self.noise_market_list.append(noise_market)
    #         self.predicted_market_list.append(predicted_market)

    # def observed_trade(self, station: Station):
    #     observed = self.predict_market.random_sample(self.aq_range, self.dp_range, self.rp_range)
    #     o_trade = station.trade(observed[0], observed[0], observed[1], observed[2])
    #     return observed, o_trade
    #
    # def predicted_ppf(self, market_index=0):
    #     return self.predicted_market_list[market_index].curve_matrix(0, 20, 0.01)

    def predicted_random(self, rounds=1000):
        power_generation, dayahead_price, realtime_price = self.predict_market.rvf(
            rounds, self.aq_range, self.dp_range, self.rp_range
        )
        return power_generation, dayahead_price, realtime_price

    def predicted_optimize(self, station: Station, recycle: BasicRecycle, power_generation, dayahead_price,
                           realtime_price):
        return DistributiveMarket.submitted_quantity_optimizer(station, recycle, power_generation, dayahead_price,
                                                               realtime_price).x

    def market_trade(
            self, x, station: Station, recycle: BasicRecycle, power_generation, dayahead_price, realtime_price
    ):
        x = numpy.asarray(x)
        if x.shape == power_generation.shape:
            pass
        else:
            x = numpy.expand_dims(x, axis=1)
            x = numpy.broadcast_to(x, power_generation.shape)
        return recycle(power_generation, x, station.trade(power_generation, x, dayahead_price, realtime_price))

    # def predicted_crps(self, market_index, aq, dp, rp):
    #     market = self.predicted_market_list[market_index]
    #     return market.faster_crps(aq, dp, rp)
    #
    # def predicted_log_score(self, market_index, aq, dp, rp):
    #     market = self.predicted_market_list[market_index]
    #     return market.faster_log_score(aq, dp, rp)

    # def index_aggregation(self, predicted_index: numpy.ndarray):
    #     l = []
    #     pg = predicted_index[:, 0]
    #     dp = predicted_index[:, 1]
    #     rp = predicted_index[:, 2]
    #     for i in range(self.market_len):
    #         for j in [pg, dp, rp]:
    #             l.append(
    #                 numpy.mean(j[:, i])
    #             )
    #             l.append(
    #                 numpy.std(j[:, i], ddof=1)
    #             )
    #     return l


if __name__ == "__main__":
    from easy_datetime.timestamp import TimeStamp

    t0 = TimeStamp.now()

    market_len = 1
    init_kwargs = {
        "aq_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 50), (1, 10), (1, 4)),
        "dp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 10), (1, 5), (1, 4)),
        "rp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 10), (1, 5), (1, 4)),
        "aq_range": (0, 50),
        "dp_range": (0, 1e+6),
        "rp_range": (0, 1e+6),
        "real_weight": numpy.full((3, market_len), 1),
        "noise_weight": numpy.full((3, market_len), 0.01),
        "market_len": market_len,
        "p_head": 10
    }

    station = Station("station", 50)
    br = PointwiseRecycle(0.5, 1.05)
    simulator = MultiMarketSimulator(**init_kwargs)

    # for _ in range(20):
    #     o, op = simulator.observed_trade(station, br)
    #     print(o)
    #     print(op)

    aq, dp, rp = simulator.predicted_random(0, 100)
    # x = simulator.predicted_optimize(station, br, aq, dp, rp)
    # print(x)
    # print(simulator.predicted_trade(x, station, br, aq, dp, rp))

    # print(
    #     simulator.predicted_crps(0, *simulator.observed_trade(station)[0])
    # )
    # print(*simulator.observed_trade(station)[0])
    #
    print(
        simulator.index_aggregation(
            numpy.asarray([simulator.predicted_crps(i, *simulator.observe()) for i in range(10)])
        )

    )
    #
    # print(
    #     [simulator.predicted_crps(i, *simulator.observe()).tolist() for i in range(3)]
    # )
    # print(simulator.predicted_ppf(0).tolist())

    print(TimeStamp.now() - t0)
