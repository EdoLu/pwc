"""
MIT License

Copyright (c) [2019] [Edoardo Luini, Philipp Arbenz]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


"""
This script contains helper function to be used in the main script "pwc.py".
"""

import numpy as np
import json
from scipy.spatial.distance import cdist
import sobol as sbl
from math import *


def ECDF(x):
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size
    return result


def firstocc(x):
    """
     first occurrence of value greater than existing value
    :param x: sorted array with unique entries
    :return: position of the first entry greater than existing value
    """
    return np.argmax(x[1:x.shape[0]] >= x[0])


def invF(p, x):
    """
    Empirical quantile function
    :param p: array of probabilities
    :param x: data, empirical points
    :return: array of quantiles (of p-level)
    """
    x_values = np.sort(np.unique(x))
    Fx = ECDF(x)(x_values)
    p = np.asarray(p).reshape((-1,))
    mFx = np.tile(Fx, p.shape[0]).reshape(p.shape[0], Fx.shape[0])
    pmFx = np.concatenate((np.array(p)[:, None], mFx), axis=1)
    vec = np.apply_along_axis(firstocc, 1, pmFx)  # .reshape(-1, 1)
    return x_values[vec]


def invG(p, max_, min_):
    """
    Empirical quantile function of pwc distribution G
    :param p: array of probabilities
    :param max_: maximum point of G support
    :param min_: minimum point of G support
    :return: array of quantiles (of p-level)
    """
    return p * (max_ - min_) + min_


def cdfG(x, max_, min_):
    """
    Distribution function of pwc distribution G
    :param x: array of quantiles
    :param max_: maximum point of G support
    :param min_: minimum point of G support
    :return: array of (cumulative) probabilities
    """
    return (x - min_) / (max_ - min_)


def argmaximum(x):
    """
    A simple argmax function that returns the position of the max; nonetheless the combination of numpy random.choice and max
    allows to select one (random) maximum if there are two or more maxima.
    :param x: array whose maximum is to be found
    :return: position of the maximum and maximum
    """
    max_ = np.max(x)
    pos_ = np.random.choice([i for i, j in enumerate(x) if j == max_])
    return [pos_, max_]


def wasserstein_oned(x, min_, max_):
    """
    wasserstein distance between a uniform (min_, max_] and a sample x in [min_, max_]
    :param x: sample in [min_, max_]
    :param min_: lower bound of uniform density
    :param max_: upper bound of uniform density
    :return: wasserstein distance value
    """

    if x.shape[0] == 0 or np.logical_and((x == min_).all(), (x == max_).all()):
        return 0

    x = np.sort(x)
    n_ = x.shape[0]
    r_ = max_ - min_
    gp_ = (np.cumsum(np.ones(n_)) - 0.5) / x.shape[0]
    gq_ = min_ + gp_ * r_
    nw_ = np.abs(x - gq_)
    floor_ = r_ / 2 * 1 / n_ - nw_
    floor_[floor_ < 0] = 0
    res = 1 / n_ * nw_ + 1 / 2 * (1 / n_ - 2 / r_ * nw_) * floor_
    pos = np.argmax(res)
    return np.asscalar(np.nansum(res)), pos


def wasserstein_oned_vert(x, min_, max_):
    """
    wasserstein distance between a uniform (min_, max_] and a sample x in [min_, max_]
    :param x: sample in [min_, max_]
    :param min_: lower bound of uniform density
    :param max_: upper bound of uniform density
    :return: wasserstein distance value
    """

    if x.shape[0] == 0 or np.logical_and((x == min_).all(), (x == max_).all()):
        return 0

    support_ = np.unique(np.sort(x))
    femp_ = np.insert(ECDF(x)(support_), 0, 0)
    support_ = np.insert(support_, 0, min_)
    support_ = np.append(support_, max_)

    r_ = max_ - min_
    midpoint = (support_[1:] + support_[:-1]) * 1/2
    b_ = (support_[1:] - support_[:-1])
    gmid_ = (midpoint - min_)/r_
    w_ = b_ * np.abs(femp_ - gmid_)
    delta = (midpoint - support_[:-1]) / r_

    floor_ = delta - w_ / b_
    adj = 1/2 * (b_ - w_ / delta) * floor_
    adj[floor_ < 0] = 0

    res = w_ + adj
    split_loc = np.argmax(res)
    return np.asscalar(np.nansum(res)), split_loc


def sample_BB(m, n, seed=None):
    """
    function to obtain the simulated-approximated distribution of the integral of
    the absolute value of a (standard) Brownian Bridge on t = [0,1]
    :param m: discretization points in [0, 1]
    :param n: number of simulation
    :param seed: random seed used
    :return: discretized distribution of the integral of the absolute value of a standard BB.
    """
    if seed is None:
        seed = 0
    np.random.seed(seed)
    dt = 1.0 / (m - 1)
    dt_sqrt = np.sqrt(dt)
    bb = np.empty((n, m), dtype=np.float32)
    bb[:, 0] = 0
    for i in range(m - 2):
        t = i * dt
        xi = np.random.randn(n) * dt_sqrt
        bb[:, i + 1] = np.abs(bb[:, i] * (1 - dt / (1 - t)) + xi)
    bb[:, -1] = 0
    bb = np.sum(bb, axis=1) / m

    return bb


"""
OT distances
"""


def ot_distance(xs, xt, ground_metric, reg):
    """
    Wrapper of sinkhorn_knoppy function for calculating OT distance, i.e. Wasserstein distance.
    :param xs: source data.
    :param xt: target data.
    :param ground_metric: cost function.
    :param reg: regularizer value (makes the optimization problem convex).
    :return: a list containing the optimal cost, the optimal map and the regularizer value.
    """
    a = np.ones((xs.shape[0], 1)) / xs.shape[0]
    b = np.ones((xt.shape[0], 1)) / xt.shape[0]
    cost_matrix = np.asarray(cdist(xs, xt, ground_metric), dtype=np.float64)
    result = sinkhorn_knopp(a, b, cost_matrix, reg)
    return result


def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-5):
    """
    Function to compute sinkhorn_knopp algorithm iteration for calculating OT distance, i.e. Wasserstein distance.
    The logic of the body of the function has been taken from https://github.com/rflamary/POT
    :param a: np.ndarray (ns,), samples weights in the source domain
    :param b: np.ndarray (nt,) or np.ndarray (nt,nbb), samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    :param M: np.ndarray (ns,nt), loss matrix
    :param reg: float, regularization term >0
    :param numItermax: int, optional, max number of iterations
    :param stopThr: float, optional, stop threshol on error (>0)
    return: OT distance value.
    """

    # we assume that no distances are null except those of the diagonal of distances
    u = np.ones((a.size, 1)) / a.size
    v = np.ones((b.size, 1)) / b.size

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while err > stopThr and cpt < numItermax:
        uprev = u
        vprev = v

        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            err = np.sum((u - uprev)**2) / np.sum(u**2) + \
                np.sum((v - vprev)**2) / np.sum(v**2)

        cpt = cpt + 1

    res = np.einsum('ik,ij,jk,ij->k', u, K, v, M)

    return res


def ot_loop(size, random_seed, m, reg, discretization_size, d, ground_metric):
    """
    Perform a loop of OT distance (i.e. Wasserstein distance) calculations between a uniform distribution and samples drawn from it.
    The logic of the body of the function has been taken from https://github.com/rflamary/POT
    :param size: positive integer, number of calculation to perform in a loop.
    :param m: positive integer, sample size of uniform samples.
    :param reg: float, regularization term >0.
    :param discretization_size: positive integer, number of element in the sobol sequence.
    :param d: positive integer, dimensionality of the problem.
    :param ground_metric: string, cost function used. Either "cityblock", "chebyshev" or "euclidean".
    :param random_seed: integer, random seed used.
    :return: array of OT distances between a uniform distribution and samples drawn from it.
    """

    stopThr = 1e-5
    numItermax = 1000

    a = np.ones((m, 1), dtype=np.float64) / m
    b = np.ones((discretization_size, 1), dtype=np.float64) / discretization_size

    xt = sobol_generator(discretization_size, d)
    w_ = np.empty(size, dtype=np.float64)
    np.random.seed(random_seed)
    for i in range(size):

        xs = np.random.uniform(size=(m, d))

        M = cdist(xs, xt, ground_metric)

        ################################################################################
        # we assume that no distances are null except those of the diagonal of distances
        u = np.ones((a.size, 1), dtype=np.float64) / a.size
        v = np.ones((b.size, 1), dtype=np.float64) / b.size

        # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
        K = np.empty(M.shape, dtype=M.dtype)
        K = np.divide(M, -reg, out=K)  # np.divide(M, -reg, out=K)
        np.exp(K, out=K)

        Kp = np.multiply(np.divide(1., a), K)
        cpt = 0
        err = 1
        while err > stopThr and cpt < numItermax:
            uprev = u
            vprev = v

            KtransposeU = np.dot(K.transpose(), u)
            v = np.divide(b, KtransposeU)
            u = 1. / np.dot(Kp, v)

            if (np.any(KtransposeU == 0)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))
                    or np.any(np.isinf(u)) or np.any(np.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break
            if cpt % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                err = np.sum((u - uprev) ** 2) / np.sum(u ** 2) + \
                      np.sum((v - vprev) ** 2) / np.sum(v ** 2)

            cpt = cpt + 1

        w_[i] = np.einsum('ik,ij,jk,ij->k', u, K, v, M)

    return w_


def parallel_ot(tup):
    # function for CPU parallel computation of OT.
    return ot_loop(*tup)


def switch_dist(argument):
    # switch function from p-norm order to the relative name of the distance.
    switcher = {
        1: "cityblock",
        2: "euclidean",
        "infty": "chebyshev"
    }
    return switcher.get(argument, "Invalid argument")


def get_unif_bounds(data):
    # dim = 1 if len(data.shape) == 1 else data.shape[1]
    if data.shape[0] == 1:
       return None, None
    x1 = np.amin(data, axis=0).reshape((1, -1))
    xn = np.amax(data, axis=0).reshape((1, -1))
    n_ = data.shape[0]
    return (n_ * x1 - xn) / (n_ - 1), (n_ * xn - x1) / (n_ - 1)


def margin_size(data):
    # Funtion that computes the size of margins of a multivariate sample (i.e. check if there are sample with duplicates)
    size = np.ones(shape=(data.shape[1],))
    for j in range(data.shape[1]):
        size[j] = np.unique(data[:, j]).shape[0]
    return size


def phi(x):
    # Cumulative distribution function for the standard normal distribution
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def sobol_generator(n, dim):
    # Wrapper to generate sobol sequence
    return sbl.i4_sobol_generate(dim_num=dim, n=n)
