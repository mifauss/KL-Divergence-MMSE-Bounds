#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import GG_example as gg


log2_p = np.linspace(-4, 8, 121)
log2_q = np.linspace(-4, 8, 121)
log2_P, log2_Q = np.meshgrid(log2_p, log2_q)

p = 2 ** log2_p
q = 2 ** log2_q
mmse_lb, crb = gg.mse_vs_p(0.0, p, q)

plt.rcParams.update({"font.size": 7})

fig3b, ax3b = plt.subplots(figsize=(3.5, 2.5))
cs = ax3b.contourf(log2_P, log2_Q, mmse_lb - crb, cmap="viridis")
ax3b.set(
    xlabel="$\log_2 \, p$", ylabel="$\log_2 \, q$",
)

cl = ax3b.contour(
    log2_P,
    log2_Q,
    mmse_lb - crb,
    cs.levels,
    colors=("white", "white", "white", "k", "k", "k", "k", "k"),
    linewidths=0.5,
)
cb = fig3b.colorbar(cs, orientation="vertical")
fig3b.savefig("contour.pdf", bbox_inches="tight")
