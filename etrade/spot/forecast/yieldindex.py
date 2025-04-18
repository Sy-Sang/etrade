#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""收益指标"""

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

# 外部模块
import numpy


# 代码块

def difference_quantile(x, y):
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    d = x - y
    return numpy.quantile(d, numpy.arange(0.01, 0.99, 0.01))


def zero_quantile(x, y):
    # x = numpy.asarray(x)
    # y = numpy.asarray(y)
    # d = numpy.sort(x - y)
    d = difference_quantile(x, y)
    return numpy.searchsorted(d, 0) / len(d)


if __name__ == "__main__":
    pass
