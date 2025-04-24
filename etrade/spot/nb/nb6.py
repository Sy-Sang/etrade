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
aq = OrdinaryGaussianKernelDistributionConstructor((0, 50), (0.1, 10), (2, 8))
dp = OrdinaryGaussianKernelDistributionConstructor((0, 10), (0.1, 10), (2, 8))
rp = OrdinaryGaussianKernelDistributionConstructor((0, 10), (0.1, 10), (2, 8))
mc = MarketConstructor(aq, dp, rp)
s = Station("station", 50)
br = PointwiseRecycle(0.5, 1.05)


class MixedMarket:
    def __init__(self, n0, n1, aq_range=(-numpy.inf, numpy.inf), dp_range=(-numpy.inf, numpy.inf),
                 rp_range=(-numpy.inf, numpy.inf)):
        self.n0 = n0
        self.n1 = n1
        self.real_market = mc.random(4)
        self.noice_market = mc.random(4)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.n0, self.n1)
        self.aq_range = aq_range
        self.dp_range = dp_range
        self.rp_range = rp_range

    def refresh(self):
        self.real_market = mc.random(4)
        self.noice_market = mc.random(4)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.n0, self.n1)

    def observed(self):
        return self.real_market.random_sample(self.aq_range, self.dp_range, self.rp_range)

    def observed_crps(self):
        data = self.observed()
        crps = self.predicted_market.faster_crps(*data)
        return crps

    def random(self, rounds=1000):
        power_generation, dayahead_price, realtime_price = self.predicted_market.rvf(rounds, self.aq_range,
                                                                                     self.dp_range, self.rp_range)
        return power_generation, dayahead_price, realtime_price

    def optimize(self, station: Station, recycle: BasicRecycle, rounds=1000):
        return self.predicted_market.submitted_quantity_optimizer(station, recycle, *self.random(rounds)).x

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


def run_once(_, aq_range, dp_range, rp_range):
    t = time.time()
    mm = MixedMarket(1000, 500, aq_range, dp_range, rp_range)
    # mm.observed()
    print(mm.observed_crps())
    result = mm.zero_quantile(s, br, 1000)
    print(f"Task done in {time.time() - t:.2f}s")
    return result


if __name__ == "__main__":
    from easy_datetime.timestamp import TimeStamp

    t0 = TimeStamp.now()

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        l = pool.map(partial(run_once, aq_range=(0, 50), dp_range=(0, 1e+6), rp_range=(0, 1e+6)), range(40))
    print(l)

    print(TimeStamp.now() - t0)
