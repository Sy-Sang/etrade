#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""配对生成器"""

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
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
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
            real_market=1000, noise_weight=1000, market_len=4,
            kernel_num=None
    ):
        self.mc = MarketConstructor(aq_constructor, dp_constructor, rp_constructor)
        self.real_weight = real_market
        self.noise_weight = noise_weight
        self.real_market = self.mc.random(market_len)
        self.noice_market = self.mc.random(market_len)
        self.predicted_market = market_hybridization(
            self.real_market, self.noice_market, self.real_weight, self.noise_weight, kernel_num
        )
        self.aq_range = aq_range
        self.dp_range = dp_range
        self.rp_range = rp_range
        self.market_len = market_len

    def refresh(self):
        self.real_market = self.mc.random(self.market_len)
        self.noice_market = self.mc.random(self.market_len)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.real_weight,
                                                     self.noise_weight)

    def replicate_noice_bandwidth_refresh(self):
        """保留噪音bandwidth的refresh"""
        aq_dist_list: list[GaussianKernelMixDistribution] = list(self.noice_market.power_generation.distributions)
        dp_dist_list: list[GaussianKernelMixDistribution] = list(self.noice_market.dayahead_price.distributions)
        rp_dist_list: list[GaussianKernelMixDistribution] = list(self.noice_market.realtime_price.distributions)
        for i in range(self.market_len):
            aq_dist_list[i] = matched_gaussian_kernel_distribution_builder(aq_dist_list[i].kernel_data())
            dp_dist_list[i] = matched_gaussian_kernel_distribution_builder(dp_dist_list[i].kernel_data())
            rp_dist_list[i] = matched_gaussian_kernel_distribution_builder(rp_dist_list[i].kernel_data())
        aq = DistributiveSeries(*aq_dist_list)
        dp = DistributiveSeries(*dp_dist_list)
        rp = DistributiveSeries(*rp_dist_list)
        self.noice_market = DistributiveMarket(aq, dp, rp)
        self.real_market = self.mc.random(self.market_len)
        self.predicted_market = market_hybridization(self.real_market, self.noice_market, self.real_weight,
                                                     self.noise_weight)

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

    def optimized_trade(self, station: Station, recycle: BasicRecycle, rounds=1000):
        x = self.optimize(station, recycle, rounds)
        power_generation, dayahead_price, realtime_price = self.real_market.rvf(
            rounds, self.aq_range, self.dp_range, self.rp_range
        )
        return self.real_market.trade_with_recycle(
            station, recycle, power_generation, dayahead_price, realtime_price, x
        ), self.real_market.trade_with_recycle(
            station, recycle, power_generation, dayahead_price, realtime_price, self.real_market.mean(rounds)[0]
        )

    def zero_quantile(self, station: Station, recycle: BasicRecycle, rounds=1000):
        opt, unopt = self.optimized_trade(station, recycle, rounds)
        return zero_quantile(opt, unopt)


def run_once(_, init_kwargs: dict, station, recycle):
    t = time.time()
    mm = MarketSimulator(**init_kwargs)
    oc = mm.observed_crps()
    # mm.replicate_noice_bandwidth_refresh()
    mm.refresh()
    # kl = mm.predicted_market.price_kl_divergence()
    kl = mm.predicted_market.ppf_difference()
    z = mm.zero_quantile(station, recycle)
    print(f"Task done in {time.time() - t:.2f}s")
    return numpy.concatenate((
        numpy.asarray(oc).reshape(-1),
        numpy.asarray(kl).reshape(-1),
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
        "real_market": 100,
        "noise_weight": 0,
        "market_len": 1,
        "kernel_num": None
    }
    s = Station("station", 50)
    br = PointwiseRecycle(0.5, 1.05)
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        l = pool.map(partial(run_once, init_kwargs=init_kwargs, station=s, recycle=br), range(400))

    with open(r"data\market_simulator_3.json", "w") as f:
        f.write(json.dumps({"data": numpy.asarray(l).tolist()}))

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        l = pool.map(partial(run_once, init_kwargs=init_kwargs, station=s, recycle=br), range(400))

    with open(r"data\market_simulator_4.json", "w") as f:
        f.write(json.dumps({"data": numpy.asarray(l).tolist()}))

    print(TimeStamp.now() - t0)

    # aq = OrdinaryGaussianKernelDistributionConstructor((0, 50), (0.1, 10), (2, 8))
    # dp = OrdinaryGaussianKernelDistributionConstructor((0, 10), (0.1, 10), (2, 8))
    # rp = OrdinaryGaussianKernelDistributionConstructor((0, 10), (0.1, 10), (2, 8))
    # s = Station("station", 50)
    # br = PointwiseRecycle(0.5, 1.05)
    # simulator = MarketSimulator(aq, dp, rp, aq_range=(0, 50), dp_range=(0, 1e+6), rp_range=(0, 1e+6), n0=1000, n1=200)
    # print(simulator.noice_market.random_sample(
    #     simulator.aq_range, simulator.dp_range, simulator.rp_range
    # ))
    # simulator.replicate_noice_bandwidth_refresh()
    # print(simulator.noice_market.random_sample(
    #     simulator.aq_range, simulator.dp_range, simulator.rp_range
    # ))
