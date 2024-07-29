# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:26:02 2024

@author: emmav
"""

from source import *

x2 = np.arange(0, 10, 1).reshape((10, 1))
y2 = np.arange(0, 10, 1).reshape((10, 1))
x = np.arange(0, 15, 1).reshape((15, 1))
y = np.arange(0, 15, 1).reshape((15, 1))

xydata = [np.hstack((x, y)), np.hstack((x2, y2))]
title = "test"
xlab = "x"
ylab = "y"
leg = (["long", "short"])
pat = save_plot_data("test", xydata, title, xlab, ylab, legend=leg)
plot_data(pat)
