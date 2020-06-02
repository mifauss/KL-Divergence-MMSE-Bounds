#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

import mmse_bounds as mb
import K_ball_example as kball
import GG_example as gg
import robustness_example as rob


###############################################################################
# Figure 1
###############################################################################

t = np.linspace(0, 3, 301)
w_vec = np.vectorize(mb.w)
w0 = w_vec(t, 0)
w1 = w_vec(t, -1)

fig1, ax1 = plt.subplots()

ax1.plot(t, w1, label="$w_1(t)$")
ax1.plot(t, w0, label="$w_0(t)$")
ax1.grid()
ax1.legend()
ax1.set(xlabel="t", title="Scaling factors of bounds")


###############################################################################
# Figure 2
###############################################################################

eps = np.linspace(0, 10, 101)
upper_bound = rob.mmse_upper_bound(0.0, 10, eps)

fig2, ax2 = plt.subplots()

ax2.plot(eps, upper_bound, label="Joint Uncertainty")
ax2.grid()
ax2.legend()
ax2.set(xlabel="\epsilon", title="Minimax MSE")


###############################################################################
# Figure 3
###############################################################################

x, y = np.mgrid[-2:2:0.01, -2:2:0.01]
pos = np.dstack((x, y))

fig3 = plt.figure(figsize=[4, 10])

Sigma_lfd = rob.sigma_lfd_2d(3, 5.0)
lfd = multivariate_normal([0.0, 0.0], Sigma_lfd)

levels1 = np.linspace(0, np.max(lfd.pdf(pos)), 10)

ax3a = fig3.add_subplot(311)
cs = ax3a.contourf(x, y, lfd.pdf(pos), levels1)
ax3a.set(xlabel="$x$", ylabel="$y$", title="LFD ($\epsilon = 0$)")
ax3a.grid()

Sigma_lfd = rob.sigma_lfd_2d(3, 0.5)
lfd = multivariate_normal([0.0, 0.0], Sigma_lfd)

levels2 = levels1[:-1]
levels2 = np.append(levels2, np.linspace(levels1[-1], np.max(lfd.pdf(pos)), 10))

ax3b = fig3.add_subplot(312)
cs = ax3b.contourf(x, y, lfd.pdf(pos), levels2)
ax3b.set(xlabel="$x$", ylabel="$y$", title="LFD ($\epsilon = 0.5$)")
ax3b.grid()

Sigma_lfd = rob.sigma_lfd_2d(3, 0)
lfd = multivariate_normal([0.0, 0.0], Sigma_lfd)

levels3 = levels2[:-1]
levels3 = np.append(levels3, np.linspace(levels2[-1], np.max(lfd.pdf(pos)), 10))

ax3c = fig3.add_subplot(313)
cs = ax3c.contourf(x, y, lfd.pdf(pos), levels3)
ax3c.set(xlabel="$x$", ylabel="$y$", title="LFD ($\epsilon = 5.0$)")
ax3c.grid()


###############################################################################
# Figure 4
###############################################################################

log2_p = np.linspace(-4, 8, 121)
log2_q = np.linspace(-4, 8, 121)
log2_P, log2_Q = np.meshgrid(log2_p, log2_q)

mmse_lb, crb = gg.mse_vs_p(0.0, 2 ** log2_p, 2 ** log2_q)

fig4 = plt.figure(figsize=[5, 7])

ax4a = fig4.add_subplot(211, projection="3d")
ax4a.plot_surface(log2_P, log2_Q, mmse_lb, cmap="viridis")
ax4a.set(
    xlabel="$\log_2 \, p$",
    ylabel="$\log_2 \, q$",
    title="Proposed MMSE bound vs p and q",
)

ax4b = fig4.add_subplot(212, projection="3d")
ax4b.plot_surface(log2_P, log2_Q, crb, cmap="viridis")
ax4b.set(
    xlabel="$\log_2 \, p$", ylabel="$\log_2 \, q$", title="Cramer-Rao bound vs p and q"
)


###############################################################################
# Figure 5
###############################################################################

fig5 = plt.figure(figsize=[5, 7])

ax5a = fig5.add_subplot(211, projection="3d")
ax5a.plot_surface(log2_P, log2_Q, mmse_lb - crb, cmap="viridis")
ax5a.set(
    xlabel="$\log_2 \, p$",
    ylabel="$\log_2 \, q$",
    title="Improvement over CRB vs p and q",
)

ax5b = fig5.add_subplot(212)
cs = ax5b.contourf(log2_P, log2_Q, mmse_lb - crb, cmap="viridis")
ax5b.set(
    xlabel="$\log_2 \, p$",
    ylabel="$\log_2 \, q$",
    title="Contour of improvement over CRB vs p and q",
)

cl = ax5b.contour(log2_P, log2_Q, mmse_lb - crb, cs.levels, colors="k")
cb = fig5.colorbar(cs, orientation="vertical")


###############################################################################
# Figure 6
###############################################################################

mmse_ub, mmse_lb, mmse_lb_apx = kball.mse_vs_K(10, 2, 100)
k = np.arange(1, 101)

fig4, ax4 = plt.subplots()
ax4.plot(k, mmse_ub, label="Upper bound")
ax4.plot(k, mmse_lb, label="Lower bound")
ax4.plot(k, mmse_lb_apx, label="Lower bound (approx.)")

ax4.grid()
ax4.legend()
ax4.set(xlabel="K", ylabel="MSE", title="Proposed MMSE Bounds vs K")


###############################################################################
# Figure 7
###############################################################################

c1 = np.linspace(1, 7, 61)
c2 = np.linspace(1, 7, 61)
C1, C2 = np.meshgrid(c1, c2)
mmse_lb = kball.mse_vs_c(c1, c2, 1)

fig5 = plt.figure()
ax5 = plt.axes(projection="3d")
ax5.plot_surface(C1, C2, mmse_lb, cmap="viridis")
ax5.set(
    xlabel="$c_1$",
    ylabel="$c_2$",
    title="Proposed MMSE bound vs center point of 2-ball",
)
