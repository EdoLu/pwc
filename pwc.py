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
Main script that contains the PWC distribution class definition.
"""
import numpy as np
import helperfunctions as hf
from collections import OrderedDict
from itertools import compress
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing as mp


class PWCDistribution:

    def __init__(self, data):
    """
    Method for defining an object belonging to the PWC distribution class. The only argument required is 'data', i.e. the observed sample.
    Ideally 'data' should be a numpy array.
    If it is not, the code authomatically converts it into a numpy array, in this case the user should check whether the resulting data is still correct.
    """

        if not isinstance(data, np.ndarray):
            Warning("data were converted to 'ndarray object' \n Please check that data shape is still correct!")
            data = np.array(data)

        # instance properties and attributes
        self.__alpha = None
        self.__ground_metric = None
        self.__p_value_correction = None
        self.__max_capacity = None
        self.__discretization_size = None
        self.__m = None

        self.__data = data
        self.__n = data.shape[0]
        self.__d = data.shape[1]
        self.__min, self.__max = hf.get_unif_bounds(data)
        self.marginal_critical = None
        self.reference_critical = None
        self.bit_array = None
        self.figure = None
        # ordered dictionary,it represents the set of all the boxes of which the pwc is composed
        self.boxes = OrderedDict()
        self.log = {"bisection": {"dimension": list(), "points": list(), "rectangles": list()},
                    "timer": 0.,
                    "n_boxes": 0,
                    "n_splits": 0}

    # property setter and getter ---------------------------------------------------------------------------------------

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, value):
        assert (0 < value < 1), "alpha should be in (0, 1)"
        self.__alpha = value

    @property
    def ground_metric(self):
        return self.__ground_metric

    @ground_metric.setter
    def ground_metric(self, value):
        assert value in [1, 2, "inf"], "%r is not a recognized ground metric" % value
        self.__ground_metric = value

    @property
    def max_capacity(self):
        return self.__max_capacity

    @max_capacity.setter
    def max_capacity(self, value):
        assert 0 < value, "max_capacity should be a positive integer"
        value = int(value) if isinstance(value, int) else value
        self.__max_capacity = value

    @property
    def p_value_correction(self):
        return self.__p_value_correction

    @p_value_correction.setter
    def p_value_correction(self, value):
        assert isinstance(value, bool), "p_value_correction should be a boolean"
        self.__p_value_correction = value

    @property
    def discretization_size(self):
        return self.__discretization_size

    @discretization_size.setter
    def discretization_size(self, value):
        assert isinstance(value, int), "discretization_size should be a positive integer"
        self.__discretization_size = value

    @property
    def m(self):
        return self.__m

    @m.setter
    def m(self, value):
        assert isinstance(value, int), "m should be a positive integer"
        self.__m = value

    @property
    def data(self):
        return self.__data

    @property
    def n(self):
        return self.__n

    @property
    def d(self):
        return self.__d

    @property
    def min(self):
        return self.__min

    @property
    def max(self):
        return self.__max

    def PWCEstimator(self, alpha=0.05, norm=1, m=None, p_value_correction=False, verbose=True):
        """
        Method that produce an admissible PWC distribution of the data. It replicates Algorithm 6.4 "Complete Algorithm".
        :param alpha: (optional) float in [0,1], significance level.
        :param norm: (optional) a value in [1, 2, or "infty"], order of the norm used as ground metric.
        :param m: (optional) positive integer, uniform sample size for reference test statistic distribution calculation. See Algorithm 5.4 "Reference test statistic critical value".
        :param p_value_correction: (optional) boolean, whether to apply the p-value correction.
        :param verbose: (optional) boolean, if true messages are displayed during execution.
        :return: PWCDistribution class, an admissible PWC distribution.
        """

        start = time.time()

        self.alpha = alpha
        self.ground_metric = norm
        self.p_value_correction = p_value_correction
        self.max_capacity = 10000 # maximum dimension size of a square matrix in terms of RAM
        self.discretization_size = int(10**self.d)
        self.m = m or int(np.log(self.n))

        self.reference_critical = self.get_reference_critical(verbose)
        self.marginal_critical = self.get_marginal_critical()

        # initiate pwc distribution with the axis-aligned minimal bounding box.
        self.boxes["Q1"] = Box(data=self.data, pwc=self, min_q=self.min, max_q=self.max)

        # data are moved into Q1 box and hence can be deleted
        self.__data = None

        if verbose:
            print("\nTesting admissibility condition...")
        while not all(box.admissible() for box in self.boxes.values()):
            self.apply_bisection(verbose=verbose)

        stop = time.time()
        self.log["n_boxes"] = len(self.boxes)
        self.log["timer"] = stop - start
        return

    def get_reference_critical(self, verbose, tolerance=0.01, confidence=0.90):
        """
	Function that estimate via simulation the 1-alpha level critical value of the test statistic distribution.
	The number of simulation is determined in such a way that there is at least 'confidence' confidence that the estimated quantile does not differ by more than 'tolerance' from the true value.
        See Section 5.2 in Meeker, Hahn and Escobar (2017). Statistical Intervals: A Guide for Practitioners and Researchers. Wiley Series in Probability and Statistics. Wiley, 2nd edition.
        """
        batch_size = 500
        wass = np.array([])
        if verbose:
            print("Computing the reference distribution...")
        level = (1 - self.alpha) * 100
        seed, jump = 12, 17
        while True:
            inputs = self.input_creator(batch_size, reg=0.01, seed=seed)
            with mp.Pool(len(inputs)) as pool:
                w_ = pool.map(hf.parallel_ot, inputs)
            w_ = np.concatenate(w_).ravel()
            wass = np.sort(np.concatenate((wass, w_), axis=0))
            seed += jump

            wp = np.percentile(wass, level)
            low = np.argmax(wass >= (1 - tolerance) * wp)
            up = np.argmax(wass >= (1 + tolerance) * wp) - 1
            mu = wass.shape[0] * (1 - self.alpha)
            sigma = np.sqrt(wass.shape[0] * (1 - self.alpha) * self.alpha)

            ci = hf.phi((up - mu) / sigma) - hf.phi((low - mu) / sigma)
            if ci >= confidence:
                break
        return wp

    def get_marginal_critical(self, m=200, size=10000, seed=None):
        """
        method to compute the alpha level quantile of the one dimensional test statistic distribution
        :param size: number of simulation and size of the distribution
        :param m: number of discretization points within the Brownian Bridge
        :param seed: pseudo-random number generator seed
        :return: the alpha level critical value
        """
        seed = seed or 2222
        np.random.seed(seed)
        bb_dist = np.sort(hf.sample_BB(m=m, n=size))
        alpha = self.alpha if not self.p_value_correction else self.alpha / self.d
        level = (1 - alpha) * 100
        critical = np.percentile(bb_dist, level)
        return critical

    def apply_bisection(self, verbose):
        # list of boxes that are not admissible (either marginal or joint)
        rect_to_bisect = [key for key in self.boxes if not (self.boxes[key].admissible())]

        for box in rect_to_bisect:

            if verbose:
                print("Splitting", box)
            split_ = self.boxes[box].get_bisection_point()

            # update log
            self.log["bisection"]["points"].append(split_["point"])
            self.log["bisection"]["dimension"].append(split_["index"])
            self.log["bisection"]["rectangles"].append(self.boxes[box].id)
            self.log["n_splits"] += 1

            # you need to declare new objects to avoid min_q_new and self.min_q being the same pointers (same for max)
            min_q_new = np.asarray(list(self.boxes[box].min_q)).astype(float).reshape((-1, self.d))
            max_q_new = np.asarray(list(self.boxes[box].max_q)).astype(float).reshape((-1, self.d))
            np.put(min_q_new, split_["index"], split_["point"])
            np.put(max_q_new, split_["index"], split_["point"])

            l_data, r_data = self.split_data(self.boxes[box].data, split_)
            # self.boxes[box].data = None

            left = Box(data=l_data, pwc=self, min_q=self.boxes[box].min_q, max_q=max_q_new)
            left.id = self.boxes[box].id + "l"

            right = Box(data=r_data, pwc=self, min_q=min_q_new, max_q=self.boxes[box].max_q)
            right.id = self.boxes[box].id + "r"

            # update boxes
            self.boxes_update(box, left, right)

        return

    def split_data(self, data, split):
        left_data = data[data[:, split["index"]] <= split["point"]].reshape((-1, self.d))
        right_data = data[data[:, split["index"]] > split["point"]].reshape((-1, self.d))
        return left_data, right_data

    def boxes_update(self, box, left_box, right_box):
        if left_box.dens == 0:
            self.boxes.update({box: right_box})
        else:
            self.boxes.update({box: left_box})
            if right_box.dens != 0:
                self.boxes["Q" + str(len(list(self.boxes)) + 1)] = right_box
        return

    def random_sample(self, sample_size, seed=None):
        seed = seed or 1212
        np.random.seed(seed)

        scheme_bins = np.array([
            [len(self.boxes[key].data) / self.n for key in self.boxes],
            [i for i in range(len(self.boxes))]
        ])
        sample_bins = np.random.choice(scheme_bins[1], size=sample_size, replace=True, p=scheme_bins[0]).astype(int)
        sample_keys = [list(self.boxes)[i] for i in sample_bins]
        return np.asarray(
            [np.random.uniform(low=self.boxes[key].min_q, high=self.boxes[key].max_q) for key in sample_keys]
        )

    def quasi_random_sample(self, sample_size):
        scheme_bins = np.array([
            [len(self.boxes[key].data) / self.n for key in self.boxes],
            [i for i in range(len(self.boxes))]
        ])
        sample_bins = np.random.choice(scheme_bins[1], size=sample_size, replace=True, p=scheme_bins[0]).astype(int)
        sample_keys = [list(self.boxes)[i] for i in sample_bins]
        a = np.array([self.boxes[key].min_q for key in sample_keys])
        b = np.array([self.boxes[key].max_q for key in sample_keys])
        x = hf.sobol_generator(sample_size, self.d)
        return a + x * (b - a)

    def pdf(self, x_):
        x_ = np.array(x_).reshape((-1, self.d))
        dens = list()
        for row in x_:
            loc = [self.boxes[j].contains(row) for j in range(len(self.boxes))]
            if not any(loc):
                dens.append(0)
            else:
                dens.append(list(compress(self.boxes, loc))[0].dens)
        dens = np.asarray(dens)
        if dens.size == 1:
            dens = np.asscalar(dens)
        return dens

    def cdf(self, x_):
        x_ = np.array(x_).reshape((-1, self.d))  # reshape x_ in case it is a scalar
        xs_ = np.dstack([x_] * len(self.boxes))
        gs = np.empty(shape=(len(self.boxes), 1))
        max_s = np.empty(shape=(x_.shape[0], x_.shape[1], len(self.boxes)))
        min_s = np.empty(shape=(x_.shape[0], x_.shape[1], len(self.boxes)))
        for s in range(len(gs)):
            key = list(self.boxes)[s]
            gs[s, :] = self.boxes[key].dens
            max_s[:, :, s] = self.boxes[key].max_q
            min_s[:, :, s] = self.boxes[key].min_q

        x_floor = np.maximum(np.subtract(xs_, min_s), 0)
        x_caged = np.minimum(x_floor, np.subtract(max_s, min_s))
        cdf = np.matmul(np.prod(x_caged, axis=1), gs)
        if cdf.size == 1:
            cdf = np.asscalar(cdf)
        return cdf

    def raw_moment(self, order):
        weights = np.asarray([box.p for box in self.boxes.values()]).reshape(-1, 1) / (order + 1)
        means = np.empty(shape=(len(self.boxes), order + 1, self.d))
        for i in range(order + 1):
            means[:, i, :] = np.asarray([box.max_q ** (order - i) * box.min_q ** i for box in self.boxes.values()]).reshape(-1, self.d)
        return np.dot(np.sum(means, axis=1).T, weights).reshape(self.d)

    def skewness(self):
        ret = (self.raw_moment(3) - 3 * self.raw_moment(1) * self.covariance().diagonal() - self.raw_moment(1)**3)
        return ret / self.covariance().diagonal()**(3/2)

    def covariance(self):
        mat_1 = np.zeros(shape=(self.d, self.d))
        mat_2 = np.zeros(shape=(1, self.d))

        for box in self.boxes:
            p_ = self.boxes[box].p
            ab = self.boxes[box].min_q + self.boxes[box].max_q
            mat_1 += p_ * np.dot(ab.T, ab) / 4
            mat_2 += p_ * ab / 2

        cov_mat = mat_1 - np.dot(mat_2.T, mat_2)
        return cov_mat

    def plot(self, showdata=False, colormap="Blues", xlim=None, ylim=None, title=True, hatch=True):
        if self.d == 2:

            xlim = xlim or [self.min[:, 0], self.max[:, 0]]
            ylim = ylim or [self.min[:, 1], self.max[:, 1]]

            figure = plt.figure()
            ax = figure.add_subplot(111)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if showdata:
                if title:
                    ax.set_title("PWC estimator and sample observation", fontsize=15)
                i = 0
                hatch_ = '\\' if hatch else None
                for key in self.boxes:
                    ax.scatter(self.boxes[key].data[:, 0], self.boxes[key].data[:, 1], c='blue', s=0.1)
                    ax.add_patch(patches.Rectangle(
                        (self.boxes[key].min_q[:, 0], self.boxes[key].min_q[:, 1]),
                        self.boxes[key].max_q[:, 0] - self.boxes[key].min_q[:, 0],  # width
                        self.boxes[key].max_q[:, 1] - self.boxes[key].min_q[:, 1],  # height
                        fill=False,
                        hatch=hatch_,
                        edgecolor="black"
                    ))
                    i += 1
            else:
                if title:
                    ax.set_title("PWC estimator", fontsize=15)
                dens_ = np.asarray([self.boxes[box].dens for box in self.boxes])
                dens_ = (dens_ - np.min(dens_)) / (np.max(dens_) - np.min(dens_))
                i = 0
                cm = plt.cm.get_cmap(colormap)
                for key in self.boxes:
                    ax.add_patch(patches.Rectangle(
                        (self.boxes[key].min_q[:, 0], self.boxes[key].min_q[:, 1]),
                        self.boxes[key].max_q[:, 0] - self.boxes[key].min_q[:, 0],  # width
                        self.boxes[key].max_q[:, 1] - self.boxes[key].min_q[:, 1],  # height
                        facecolor=cm(dens_[i]),
                        edgecolor=cm(dens_[i])  # "grey" # "black"
                        ))
                    i += 1
            return ax
        else:
            print("This method works only when dimension is 2!")
            return

    def input_creator(self, size, reg, cost=4, seed=22):
        """
	Helper method that creates a list of input to parallelize the computation of OT distance
        """
        num_threads = mp.cpu_count() // cost
        sizes = [size // num_threads] * num_threads
        sizes[-1] = sizes[-1] + size % num_threads
        seeds = list(range(seed, seed + num_threads))
        inputs = list()
        for i in range(len(seeds)):
            list_ = [sizes[i], seeds[i], self.m, reg, self.discretization_size, self.d, hf.switch_dist(self.ground_metric)]
            inputs.append(list_)
        return inputs


class Box:
    """
    Box class is the object representing a PWC distribution hyperrectangle partitioning the domain.
    """
    def __init__(self, pwc, data, min_q, max_q, id_=""):

        self.__data = data
        self.__min_q = min_q
        self.__max_q = max_q
        self.__id = id_

        self.p = self.data.shape[0] / pwc.n
        self.dens = self.p / self.volume()

        self.alpha = pwc.__dict__['_PWCDistribution__alpha']
        self.max_capacity = pwc.__dict__['_PWCDistribution__max_capacity']
        self.ground_metric = pwc.__dict__['_PWCDistribution__ground_metric']
        self.discretization_size = pwc.__dict__['_PWCDistribution__discretization_size']
        self.m = pwc.__dict__['_PWCDistribution__m']

        self.split_loc = np.ones(shape=(self.data.shape[1]), dtype=int)
        self.marginal_wasserstein = self.get_marginal_wasserstein()
        self.marginal_critical = self.get_marginal_critical(pwc.marginal_critical)

        self.joint_wasserstein = self.get_joint_wassersten()
        self.joint_critical = self.get_joint_critical(pwc.reference_critical)

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            Warning("data were converted to 'ndarray object!")
            value = np.array(value)
        self.__data = value

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        value = value or ""
        self.__id = value

    @property
    def min_q(self):
        return self.__min_q

    @min_q.setter
    def min_q(self, value):
        assert np.all(value <= self.__max_q), "min_q should be less than or equal to max_q!"
        self.__min_q = value

    @property
    def max_q(self):
        return self.__max_q

    @max_q.setter
    def max_q(self, value):
        assert np.all(value >= self.__min_q), "max_q should be greater than or equal to min_q!"
        self.__max_q = value

    def marginal_admissible(self):
        return np.all(self.marginal_wasserstein <= self.marginal_critical)

    def admissible(self):
        # test whether a box is marginal admissible, if it is then test if it is joint admissible
        if not self.marginal_admissible():
            return False
        else:
            return self.joint_wasserstein <= self.joint_critical

    def get_marginal_critical(self, value):
        if self.data.shape[0] == 0:
            return np.zeros(shape=self.min_q.shape[1])

        size_margins = hf.margin_size(self.data)
        if np.all(size_margins > 1):
            return ((value * (self.max_q - self.min_q)) / np.sqrt(self.data.shape[0])).reshape((self.data.shape[1],))
        else:
            ret = np.empty(shape=self.data.shape[1])
            for j in range(self.data.shape[1]):
                ret[j] = (value * (self.max_q[:, j] - self.min_q[:, j])) / np.sqrt(size_margins[j])
            return ret

    def get_joint_critical(self, reference_critical):

        if not self.marginal_admissible():
            return 0
        if self.data.shape[0] == 0:
            return 0

        critical_ = self.order_of_convergence_adjustment(reference_critical)
        return critical_

    def order_of_convergence_adjustment(self, critical_):
        d_ = self.data.shape[1]
        n_ = self.data.shape[0]
        if d_ > 2:
            critical_ = critical_ * (n_ / self.m) ** (-1 / d_)
        elif d_ == 2:
            critical_ = critical_ * ((np.log(n_) / float(n_)) ** 0.5) / ((np.log(self.m) / float(self.m)) ** 0.5)
        return critical_

    def get_marginal_wasserstein(self):
        if self.data.shape[0] == 0:
            return np.zeros(shape=self.min_q.shape[1])

        wasserstein_m = np.ones(shape=self.data.shape[1])
        for j in range(self.data.shape[1]):
            x = self.data[:, j]
            min_ = self.min_q[:, j]
            max_ = self.max_q[:, j]

            if np.unique(x).shape[0] == 1:
                x = np.unique(x)
                ret, self.split_loc[j] = hf.wasserstein_oned_vert(x, min_, max_)
                wasserstein_m[j] = ret
            else:
                ret, self.split_loc[j] = hf.wasserstein_oned_vert(x, min_, max_)
                wasserstein_m[j] = ret

        return wasserstein_m

    def get_joint_wassersten(self):
        if not self.marginal_admissible():
            return 0
        if self.data.shape[0] == 0:
            return 0

        reg = 0.01
        ground_metric = hf.switch_dist(self.ground_metric)
        # transform sample data
        Utilde = (self.data - self.min_q) / (self.max_q - self.min_q)

        if np.any(hf.margin_size(Utilde) == 1):
            return 0
        elif Utilde.shape[0] >= self.max_capacity:
            sbl_sample = hf.sobol_generator(self.discretization_size, self.data.shape[1])
            ret_vec = np.empty(1)
            for i in range(ret_vec.shape[0]):
                subsample = Utilde[np.random.choice(Utilde.shape[0], self.max_capacity, replace=True), :]
                ret_vec[i] = hf.ot_distance(xs=subsample, xt=sbl_sample, reg=reg, ground_metric=ground_metric)
            ret = np.mean(ret_vec)
        else:
            sbl_sample = hf.sobol_generator(self.discretization_size, self.data.shape[1])
            ret = hf.ot_distance(xs=Utilde, xt=sbl_sample, ground_metric=ground_metric, reg=reg)
        # correct for avoiding numerical error
        if ret <= 10**(-6):
            ret = 0
        return ret

    def get_bisection_point(self):

        dim_to_split = self.get_dim_to_split()
        data = self.data[:, dim_to_split]
        min_ = self.min_q[:, dim_to_split]
        max_ = self.max_q[:, dim_to_split]

        data = np.sort(np.append(np.unique(data), (min_, max_)))
        candidates = 0.5 * (data[:-1] + data[1:])
        if (data.shape[0] - 2) > 1:
            lowerb, upperb = hf.get_unif_bounds(data[1:-1])
            candidates[0] = max(candidates[0], lowerb)
            candidates[-1] = min(candidates[-1], upperb)
        split_point = candidates[self.split_loc[dim_to_split]]
        res = {"index": dim_to_split, "point": split_point}
        return res

    def get_dim_to_split(self):
        w_list = self.marginal_wasserstein
        std_w_list = w_list / (self.max_q - self.min_q).reshape(w_list.shape[0], )
        return hf.argmaximum(np.sqrt(self.data.shape[0]) * std_w_list)[0]

    def volume(self):
        volume = float(np.prod(self.max_q - self.min_q)) if float(np.prod(self.max_q - self.min_q)) > 1e-3 else 1
        return volume

    def random_sample(self, sample_size=1, seed=None):
        seed = seed or 1212
        np.random.seed(seed)
        runif = np.random.uniform(size=self.data.shape[1] * sample_size).reshape(sample_size, self.data.shape[1])
        return self.min_q + runif * (self.max_q - self.min_q)

    def quasi_random_sample(self, sample_size=1):
        sobol_unif = hf.sobol_generator(sample_size, self.data.shape[1])
        np.random.shuffle(sobol_unif)
        return self.min_q + sobol_unif * (self.max_q - self.min_q)

    def contains(self, x):
        """Return true if a point is inside the rectangle."""
        return np.all(x >= self.min_q) * np.all(x <= self.max_q)

