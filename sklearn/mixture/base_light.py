"""Base class for mixture models."""

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# Modified further by Mark Edgington <edgimar@gmail.com>

from __future__ import print_function

import warnings
from abc import ABCMeta, abstractmethod
from time import time

import numpy as np

from .. import cluster
from ..base import BaseEstimator
from ..base import DensityMixin
from ..externals import six
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state
from ..utils.extmath import logsumexp




def _check_shape(param, param_shape, name):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : string
    """
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError("The parameter '%s' should have the shape of %s, "
                         "but got %s" % (name, param_shape, param.shape))


def _check_X(X, n_components=None, n_features=None):
    """Check the input data X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    n_components : int

    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32])
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.  In contrast to the
    BaseMixture class in base.py, this version doesn't assume that EM will be
    used as an estimator.
    """

    def __init__(self, reg_covar, random_state):
        """
        Parameters
        ----------
        reg_covar : float > 0
            a number indicating how large the covariance used for
            regularization should be.  Used by subclasses.

        """
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        if self.reg_covar < 0.:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)

        # Check all the parameters values of the derived class
        self._check_parameters(X)

    @abstractmethod
    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        This method should validate all parameters that have been stored as
        instance variables in the derived class.  If there are invalid
        parameter values, a ValueError should be raised which explains the
        problem.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        """
        pass

    #def _initialize_parameters(self, X):
    #    """Initialize the model parameters.

    #    Parameters
    #    ----------
    #    X : array-like, shape  (n_observations, n_features)
    #    """
    #    n_observations = X.shape[0]


    @abstractmethod
    def fit(self, X, y=None):
        """Estimate model parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        pass

    def fit_ORIG(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        The method fit the model `n_init` times and set the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        X = _check_X(X, self.n_components)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_log_likelihood = -np.infty
        self.converged_ = False

        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X)
            current_log_likelihood, resp = self._e_step(X)

            for n_iter in range(self.max_iter):
                prev_log_likelihood = current_log_likelihood

                self._m_step(X, resp)
                current_log_likelihood, resp = self._e_step(X)
                change = current_log_likelihood - prev_log_likelihood
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(current_log_likelihood)

            if current_log_likelihood > max_log_likelihood:
                max_log_likelihood = current_log_likelihood
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn('Initialization %d did not converged. '
                          'Try different init parameters, '
                          'or increase n_init, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter

        return self

    @abstractmethod
    def _check_is_fitted(self):
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    @abstractmethod
    def _set_parameters(self, params):
        pass

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given X.
        """
        return self.score_samples(X).mean()

    def predict(self, X, y=None):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data per each component.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])
        _, _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_features, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    @abstractmethod
    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass

    @abstractmethod
    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        pass

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_prob : array, shape (n_samples, n_components)
            log p(X|Z) + log weights

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, weighted_log_prob, log_resp

