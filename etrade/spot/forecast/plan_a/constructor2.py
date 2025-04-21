#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""构造器2"""

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

# 项目模块
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution, \
    SkewNormalDistribution, LogNormalDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.kernel.gaussian import \
    GaussianKernelMixDistribution
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.kernel2 import KernelMixDistribution
from data_utils.stochastic_utils.vdistributions.tools.clip import ClampedDistribution

from etrade.spot.trader import Station
from etrade.spot.market.recycle import BasicRecycle, PointwiseRecycle
from etrade.spot.forecast.market import DistributiveSeries, DistributiveMarket

from etrade.spot.forecast.plan_a.constructor import AbstractDistributionConstructor

# 外部模块
import numpy


# 代码块

class KLDivergenceConstructor(AbstractDistributionConstructor):

    def __init__(self, dist: AbstractDistribution, kl_divergence):
        self.distribution = dist
        self.kl_divergence = kl_divergence


if __name__ == "__main__":
    pass
