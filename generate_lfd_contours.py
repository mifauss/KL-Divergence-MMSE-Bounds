#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

import robustness_example as rob


x, y = np.mgrid[-2:2:0.01, -2:2:0.01]
pos = np.dstack((x, y))

plt.rcParams.update({"font.size": 7})


Sigma_lfd = rob.sigma_lfd_2d(3, 5.0)
lfd = multivariate_normal([0.0, 0.0], Sigma_lfd)

levels1 = np.linspace(0, np.max(lfd.pdf(pos)), 10)

fig3, ax3 = plt.subplots(figsize=(3.5, 2.0))
cs = ax3.contourf(x, y, lfd.pdf(pos), levels1)
ax3.set(xlabel="$x$", ylabel="$y$")
ax3.grid()
fig3.savefig("lfd_contour_50.pdf", bbox_inches="tight")


Sigma_lfd = rob.sigma_lfd_2d(3, 0.5)
lfd = multivariate_normal([0.0, 0.0], Sigma_lfd)

levels2 = levels1[:-1]
levels2 = np.append(levels2, np.linspace(levels1[-1], np.max(lfd.pdf(pos)), 10))

fig2, ax2 = plt.subplots(figsize=(3.5, 2.0))
cs = ax2.contourf(x, y, lfd.pdf(pos), levels2)
ax2.set(xlabel="$x$", ylabel="$y$")
ax2.grid()
fig2.savefig("lfd_contour_05.pdf", bbox_inches="tight")


Sigma_lfd = rob.sigma_lfd_2d(3, 0)
lfd = multivariate_normal([0.0, 0.0], Sigma_lfd)

levels3 = levels2[:-1]
levels3 = np.append(levels3, np.linspace(levels2[-1], np.max(lfd.pdf(pos)), 10))

fig1, ax1 = plt.subplots(figsize=(3.5, 2.0))
cs = ax1.contourf(x, y, lfd.pdf(pos), levels3)
ax1.set(xlabel="$x$", ylabel="$y$")
ax1.grid()
fig1.savefig("lfd_contour_0.pdf", bbox_inches="tight")
