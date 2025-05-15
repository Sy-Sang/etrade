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
        counter = 1
        for i in range(self.power_generation.len):
            pyplot.subplot(3, 1, counter)

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
        """带有相关性的随机样本"""
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

    def market_trade(self, station: Station, recycle: Recycle, x, num=1000):
        aq, dp, rp = self.rvf(num)
        return self.trade_with_recycle(station, recycle, aq, dp, rp, x)

    def power_generation_optimizer(self, station: Station, recycle: Recycle, q_min=0, q_max=None, num=1000):
        q_max = station.max_power if q_max is None else q_max
        power_generation, dayahead_price, realtime_price = self.rvf(num)

        def f(x):
            return numpy.mean(
                self.trade_with_recycle(station, recycle, power_generation, dayahead_price,
                                        realtime_price, x)
            ) * -1

        result = differential_evolution(
            f,
            [(q_min, q_max)] * self.power_generation.len,
            strategy='best1bin',  # 变异策略
            # popsize=10,  # 种群大小（默认15，越小越快但精度低）
            # mutation=(0.5, 1.0),  # 变异范围
            # recombination=0.9,  # 交叉概率
            tol=1e-5,
            polish=True,  # 自动调用L-BFGS-B精修
            # workers=-1,  # 多核并行
            # updating='deferred',  # 提升并行效率
        )
        return result

    def faster_power_generation_optimizer(self, station: Station, recycle: Recycle, q_min=0, q_max=None, num=1000):
        q_min = 0 if q_min is None else q_min
        q_max = station.max_power if q_max is None else q_max
        power_generation, dayahead_price, realtime_price = self.rvf(num)
        mean = self.mean(num)

        def f(x):
            return numpy.mean(
                self.trade_with_recycle(station, recycle, power_generation, dayahead_price,
                                        realtime_price, x)
            ) * -1

        result = minimize(f, mean[0], bounds=[(q_min, q_max)] * self.power_generation.len)
        return result

    @classmethod
    def submitted_quantity_optimizer(cls, station: Station, recycle: Recycle, aq, dp, rp, q_min=0, q_max=None):
        """sq优化器"""
        q_max = station.max_power if q_max is None else q_max

        def f(x):
            return numpy.mean(
                cls.trade_with_recycle(station, recycle, aq, dp, rp, x)
            ) * -1

        result = differential_evolution(f, [(q_min, q_max)] * len(aq), strategy='best1bin',
                                        mutation=(0.5, 1), recombination=0.7,
                                        popsize=15, maxiter=1000, tol=1e-6)
        return result

    def crps(self, aq, dp, rp):
        """crps"""
        aq = numpy.asarray(aq).reshape(-1)
        dp = numpy.asarray(dp).reshape(-1)
        rp = numpy.asarray(rp).reshape(-1)
        l = [[], [], []]
        for i in range(self.power_generation.len):
            l[0].append(
                crps(self.power_generation.distributions[i], aq[i])
            )
            l[1].append(
                crps(self.dayahead_price.distributions[i], dp[i])
            )
            l[2].append(
                crps(self.realtime_price.distributions[i], rp[i])
            )
        return numpy.asarray(l)

    def faster_crps(self, aq, dp, rp):
        def private_crps(dist: AbstractDistribution, value, num_points=100):
            domain_min, domain_max = dist.domain()
            x = numpy.linspace(domain_min, domain_max, num_points)
            fx = (dist.cdf(x) - numpy.where(x >= value, 1, 0)) ** 2
            return numpy.trapz(fx, x)

        aq = numpy.asarray(aq).reshape(-1)
        dp = numpy.asarray(dp).reshape(-1)
        rp = numpy.asarray(rp).reshape(-1)
        l = [[], [], []]
        for i in range(self.power_generation.len):
            l[0].append(
                private_crps(self.power_generation.distributions[i], aq[i])
            )
            l[1].append(
                private_crps(self.dayahead_price.distributions[i], dp[i])
            )
            l[2].append(
                private_crps(self.realtime_price.distributions[i], rp[i])
            )
        return numpy.asarray(l)

    def curve_matrix(self, curve_index=0):
        m = []
        for i in range(self.market_len):
            column = [
                self.power_generation.distributions[i].curves(10, 0.1)[curve_index][:, 1],
                self.dayahead_price.distributions[i].curves(10, 0.1)[curve_index][:, 1],
                self.realtime_price.distributions[i].curves(10, 0.1)[curve_index][:, 1],
            ]
            m.append(column)
        return numpy.asarray(m).reshape(-1)

    def faster_log_score(self, aq, dp, rp):
        aq = numpy.asarray(aq).reshape(-1)
        dp = numpy.asarray(dp).reshape(-1)
        rp = numpy.asarray(rp).reshape(-1)
        l = [[], [], []]
        for i in range(self.market_len):
            l[0].append(
                numpy.log(self.power_generation.distributions[i].pdf(aq[i]))
            )
            l[1].append(
                numpy.log(self.dayahead_price.distributions[i].pdf(dp[i]))
            )
            l[2].append(
                numpy.log(self.realtime_price.distributions[i].pdf(rp[i]))
            )
        return numpy.asarray(l)



    # def price_kl_divergence(self):
    #     """价格的kl散度"""
    #     l = []
    #     for i in range(self.power_generation.len):
    #         l.append(
    #             js_divergence_continuous(
    #                 self.dayahead_price.distributions[i],
    #                 self.realtime_price.distributions[i]
    #             )
    #         )
    #     return numpy.asarray(l)
    #
    # def quantile_rmse_matrix(self):
    #     """kl散度矩阵"""
    #     m = []
    #     for r in range(3):
    #         row = []
    #         for c in range(self.power_generation.len):
    #             row.append(
    #                 [quantile_RMSE(self.map[r].distributions[c], self.map[r].distributions[i]) for i in
    #                  range(self.power_generation.len)]
    #             )
    #         m.append(row)
    #     return m
    #
    # def pdf_difference(self, num=100):
    #     v = []
    #     for i in range(self.market_len):
    #         dayahead_domain = self.dayahead_price.distributions[i].domain()
    #         realtime_domain = self.realtime_price.distributions[i].domain()
    #         domain_min = numpy.min([dayahead_domain[0], realtime_domain[0]])
    #         domain_max = numpy.max([dayahead_domain[1], realtime_domain[1]])
    #         x = numpy.linspace(domain_min, domain_max, num)
    #         v.append(
    #             numpy.sum(
    #                 self.dayahead_price.distributions[i].pdf(x) - self.realtime_price.distributions[i].pdf(x) ** 2
    #             )
    #         )
    #     return v
    #
    # def ppf_difference(self, num=20):
    #     def to_positive(x):
    #         delta = 1 - numpy.min(x)
    #         return x + delta
    #
    #     def log_diff(x):
    #         return numpy.diff(numpy.log(to_positive(x)))
    #
    #     def zscore(x):
    #         std = numpy.std(x, ddof=1)
    #         if std == 0:
    #             raise Exception("[error]: std=0")
    #         else:
    #             return (x - numpy.mean(x)) / numpy.std(x, ddof=1)
    #
    #     v = numpy.empty((self.market_len, (num - 0) * 3))
    #     for i in range(self.market_len):
    #         power_ppf = self.power_generation.distributions[i].curves(num, 0.01)[0][:, 1]
    #         dayahead_ppf = self.dayahead_price.distributions[i].curves(num, 0.01)[0][:, 1]
    #         realtime_ppf = self.realtime_price.distributions[i].curves(num, 0.01)[0][:, 1]
    #         v[i] = numpy.concatenate((
    #             # log_diff(power_ppf),
    #             # log_diff(dayahead_ppf),
    #             # log_diff(realtime_ppf)
    #             power_ppf, dayahead_ppf, realtime_ppf
    #             # dayahead_ppf, realtime_ppf
    #         ))
    #     return v


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
    from matplotlib import pyplot

    d = DistributiveSeries(NormalDistribution(0, 1), NormalDistribution(100, 10))
    pyplot.scatter(*d.correlated_rvf(10, [1, 1]))
    pyplot.show()
    # print(d.rvf(10))
