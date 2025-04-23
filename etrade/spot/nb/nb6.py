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

# 项目模块
from etrade.spot.forecast.market import DistributiveSeries, DistributiveMarket
from etrade.spot.forecast.plan_a.constructor import *
from etrade.spot.forecast.yieldindex import zero_quantile

# 外部模块
import numpy

# 代码块
aq = OrdinaryGaussianKernelDistributionConstructor((0, 50), (0.1, 10), (2, 8))
dp = OrdinaryGaussianKernelDistributionConstructor((0, 10), (0.1, 10), (2, 8))
rp = OrdinaryGaussianKernelDistributionConstructor((0, 10), (0.1, 10), (2, 8))
mc = MarketConstructor(aq, dp, rp)
s = Station("station", 50)
br = PointwiseRecycle(0.5, 1.05)


class MixedMarket:
    def __init__(self, n0, n1):
        self.n0 = n0
        self.n1 = n1
        self.real_market = mc.random(4)
        self.noice_market = mc.random(4)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.n0, self.n1)

    def refresh(self):
        self.real_market = mc.random(4)
        self.noice_market = mc.random(4)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.n0, self.n1)

    def observed(self):
        return self.real_market.random_sample()

    def optimize(self, station: Station, recycle: BasicRecycle, rounds=1000):
        return self.predicted_market.power_generation_optimizer(station, recycle, num=rounds).x

    def faster_optimize(self, station: Station, recycle: BasicRecycle, rounds=1000):
        return self.predicted_market.faster_power_generation_optimizer(station, recycle, num=rounds).x

    def zero_quantile(self, station: Station, recycle: BasicRecycle, rounds=1000, faster=False):
        if faster:
            x = self.faster_optimize(station, recycle, rounds)
        else:
            x = self.optimize(station, recycle, rounds)
        return zero_quantile(
            self.real_market.market_trade(station, recycle, x, num=rounds),
            self.real_market.market_trade(station, recycle, self.real_market.mean(rounds)[0], num=rounds)
        )


def run_once(_):
    t = time.time()
    mm = MixedMarket(1000, 500)
    mm.observed()
    result = mm.zero_quantile(s, br, 1000, faster=False)
    print(f"Task done in {time.time() - t:.2f}s")
    return result


if __name__ == "__main__":
    from easy_datetime.timestamp import TimeStamp

    t0 = TimeStamp.now()

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        l = pool.map(run_once, range(40))  # 你原来是 range(20)
    print(l)

    print(TimeStamp.now() - t0)
