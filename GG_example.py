#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma

import mmse_bounds


def kl_div_GG(p):
    if p == 2.0:
        return 0.0
    else:
        return (
            np.log(p / np.sqrt(2))
            + 0.5 * np.log(gamma(3 / p) / gamma(1 / p))
            + np.log(gamma(1 / 2) / gamma(1 / p))
            + 0.5
            - 1 / p
        )


def kl_div_joint(p, q):
    return kl_div_GG(p) + kl_div_GG(q)


def fisher_inf_GG(p, var):
    return ((p ** 2) * gamma(3 / p) * gamma(2 - 1 / p)) / (var * gamma(1 / p) ** 2)


def mse_vs_p(snr_db, p, q):
    snr = 10 ** (snr_db / 10)
    P, Q = np.meshgrid(p, q)

    s_X = 1
    s_N = 1 / snr
    Sigma0 = np.array([[s_X, s_X], [s_X, s_X + s_N]])

    mmse_lb = np.zeros_like(P)
    crb = np.zeros_like(P)

    for i in range(P.size):
        pi = P.flat[i]
        qi = Q.flat[i]

        # KL bound
        eps = kl_div_joint(pi, qi)
        mmse_lb.flat[i] = mmse_bounds.lower(Sigma0, 1, 1, eps)[0]

        # CRB
        if pi > 0.5 and qi > 0.5:
            J_X = fisher_inf_GG(pi, s_X)
            J_N = fisher_inf_GG(qi, s_N)
            crb.flat[i] = 1 / (J_X + J_N)
        else:
            crb.flat[i] = 0.0

    return mmse_lb, crb
