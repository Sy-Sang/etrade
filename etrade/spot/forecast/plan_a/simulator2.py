#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""模拟器2"""

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
from etrade.spot.forecast.market import DistributiveSeries, DistributiveMarket
from etrade.spot.forecast.plan_a.constructor import *
from etrade.spot.forecast.yieldindex import zero_quantile

from etrade.spot.forecast.plan_a.simulator import MarketSimulator

# 外部模块
import numpy


# 代码块

class WeightGaussianMarketSimulator(MarketSimulator):
    """
    按核权重添加噪音的市场模拟器
    """

    def __init__(
            self,
            aq_constructor: OrdinaryGaussianKernelDistributionConstructor,
            dp_constructor: OrdinaryGaussianKernelDistributionConstructor,
            rp_constructor: OrdinaryGaussianKernelDistributionConstructor,
            aq_range=(-numpy.inf, numpy.inf),
            dp_range=(-numpy.inf, numpy.inf),
            rp_range=(-numpy.inf, numpy.inf),
            real_weight=numpy.ones((3, 4)),
            noise_weight=numpy.ones((3, 4)),
            market_len=4
    ):
        # super().__init__(aq_constructor, dp_constructor, rp_constructor, aq_range, dp_range, rp_range, real_market,
        #                  noise_weight, market_len, kernel_num)
        self.mc = MarketConstructor(aq_constructor, dp_constructor, rp_constructor)
        self.real_weight = real_weight
        self.noise_weight = noise_weight
        self.real_market = self.mc.random(market_len)
        self.noise_market = self.mc.random(market_len)
        self.predicted_market = market_hybridization_by_weight(
            self.real_market, self.noise_market, self.real_weight, self.noise_weight
        )
        self.aq_range = aq_range
        self.dp_range = dp_range
        self.rp_range = rp_range
        self.market_len = market_len

    def refresh(self):
        self.real_market = self.mc.random(self.market_len)
        self.noise_market = self.mc.random(self.market_len)
        self.predicted_market = market_hybridization_by_weight(
            self.real_market, self.noise_market, self.real_weight, self.noise_weight
        )


def run_once(_, init_kwargs: dict, station, recycle, rounds=1000):
    t = time.time()
    mm = WeightGaussianMarketSimulator(**init_kwargs)
    oc = mm.observed_crps()
    # mm.replicate_noice_bandwidth_refresh()
    mm.refresh()
    # kl = mm.predicted_market.price_kl_divergence()
    kl = mm.predicted_market.ppf_difference(5)
    # opt, z = mm.alpha_quantile(station, recycle, 0.2)
    x = mm.optimize(station, recycle, rounds)
    opt, unopt = mm.predicted_market_trade(x, station, recycle, rounds)
    ropt, runopt = mm.real_market_trade(x, station, recycle, rounds)
    z = mm.alpha_quantile(ropt, runopt, 0.5)

    print(f"Task done in {time.time() - t:.2f}s")
    return numpy.concatenate((
        numpy.asarray(oc).reshape(-1),
        numpy.asarray(kl).reshape(-1),
        numpy.quantile(
            numpy.asarray(opt) - numpy.asarray(unopt),
            EasyFloat.frange(0.1, 0.9, 0.1, True)
        ),
        numpy.atleast_1d(z).reshape(-1)
    ))


if __name__ == "__main__":
    from easy_datetime.timestamp import TimeStamp

    t0 = TimeStamp.now()

    init_kwargs = {
        "aq_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 50), (1, 10), (1, 8)),
        "dp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 10), (1, 10), (1, 8)),
        "rp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 10), (1, 10), (1, 8)),
        "aq_range": (0, 50),
        "dp_range": (0, 1e+6),
        "rp_range": (0, 1e+6),
        # "real_market": 100,
        "noise_weight": numpy.full((3, 1), 0.1),
        "market_len": 1,
        # "kernel_num": None
    }

    init_kwargs_1 = {
        "aq_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 50), (1, 10), (1, 8)),
        "dp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 10), (1, 10), (1, 8)),
        "rp_constructor": OrdinaryGaussianKernelDistributionConstructor((0, 10), (1, 10), (1, 8)),
        "aq_range": (0, 50),
        "dp_range": (0, 1e+6),
        "rp_range": (0, 1e+6),
        # "real_market": 100,
        "noise_weight": numpy.full((3, 1), 0.1),
        "market_len": 1,
        # "kernel_num": None
    }

    s = Station("station", 50)
    br = PointwiseRecycle(0.5, 1.05)
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        l = pool.map(partial(run_once, init_kwargs=init_kwargs, station=s, recycle=br), range(3000))

    with open(r"data\market_simulator_5.json", "w") as f:
        f.write(json.dumps({"data": numpy.asarray(l).tolist()}))

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        l = pool.map(partial(run_once, init_kwargs=init_kwargs_1, station=s, recycle=br), range(600))

    with open(r"data\market_simulator_6.json", "w") as f:
        f.write(json.dumps({"data": numpy.asarray(l).tolist()}))

    print(TimeStamp.now() - t0)
