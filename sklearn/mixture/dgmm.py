"""
Dynamic Gaussian Mixture Model

This is an implementation of a Gaussian Mixture Model whose size and parameters
are dynamically changed via the Dynamic Gaussian Mixture Estimation (DGME)
algorithm.

Various parts of this module have been borrowed from the gaussian_mixture.py module.
"""

# Author: Mark Edgington <edgimar@gmail.com>

from __future__ import division
import numpy as np

from scipy import linalg

from .base_light import BaseMixture, _check_shape
from ..externals.six.moves import zip
from ..utils import check_array
from ..utils.validation import check_is_fitted


###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0)) or
            any(np.greater(weights, 1))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1 - np.sum(weights)), 0.0):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means


def _check_covariance_matrix(covariance, covariance_type):
    """Check a covariance matrix is symmetric and positive-definite."""
    if (not np.allclose(covariance, covariance.T) or
            np.any(np.less_equal(linalg.eigvalsh(covariance), .0))):
        raise ValueError("'%s covariance' should be symmetric, "
                         "positive-definite" % covariance_type)


def _check_covariance_positivity(covariance, covariance_type):
    """Check a covariance vector is positive-definite."""
    if np.any(np.less_equal(covariance, 0.0)):
        raise ValueError("'%s covariance' should be "
                         "positive" % covariance_type)


def _check_covariances_full(covariances, covariance_type):
    """Check the covariance matrices are symmetric and positive-definite."""
    for k, cov in enumerate(covariances):
        _check_covariance_matrix(cov, covariance_type)


def _check_covariances(covariances, covariance_type, n_components, n_features):
    """Validate user provided covariances.

    Parameters
    ----------
    covariances : array-like,
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    covariances : array
    """
    covariances = check_array(covariances, dtype=[np.float64, np.float32],
                              ensure_2d=False,
                              allow_nd=covariance_type is 'full')

    covariances_shape = {'full': (n_components, n_features, n_features),
                         'tied': (n_features, n_features),
                         'diag': (n_components, n_features),
                         'spherical': (n_components,)}
    _check_shape(covariances, covariances_shape[covariance_type],
                 '%s covariance' % covariance_type)

    check_functions = {'full': _check_covariances_full,
                       'tied': _check_covariance_matrix,
                       'diag': _check_covariance_positivity,
                       'spherical': _check_covariance_positivity}
    check_functions[covariance_type](covariances, covariance_type)

    return covariances

#
# Gaussian mixture parameters estimators used in M-Step
# Gaussian mixture probability estimators used in E-Step


def _estimate_log_gaussian_prob_full(X, means, covariances):
    """Estimate the log Gaussian probability for 'full' covariance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    covariances : array-like, shape (n_components, n_features, n_features)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components))
    for k, (mu, cov) in enumerate(zip(means, covariances)):
        try:
            cov_chol = linalg.cholesky(cov, lower=True)
        except linalg.LinAlgError:
            raise ValueError("The algorithm has diverged because of too "
                             "few samples per components. "
                             "Try to decrease the number of components, or "
                             "increase reg_covar.")
        cv_log_det = 2. * np.sum(np.log(np.diagonal(cov_chol)))
        cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                         lower=True).T
        log_prob[:, k] = - .5 * (n_features * np.log(2. * np.pi) +
                                 cv_log_det +
                                 np.sum(np.square(cv_sol), axis=1))
    return log_prob


class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Parameters
    ----------
    covariance_type : {'full', 'tied', 'diag', 'spherical'},
        defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::
        'full' (each component has its own general covariance matrix).
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'spherical' (each component has its own single variance),

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init: array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.

    covariances_init: array-like, optional.
        The user-provided initial covariances, defaults to None.
        If it None, covariances are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state: RandomState or an int seed, defaults to None.
        A random number generator instance.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    Attributes
    ----------
    weights_ : array, shape (n_components,)
        The unnormalized weights of each mixture components.
        `weights_` will not exist before a call to fit.

    means_ : array, shape (n_components, n_features)
        The mean of each mixture component.
        `means_` will not exist before a call to fit.

    covariances_ : array
        The covariance of each mixture component.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
        `covariances_` will not exist before a call to fit.

    """

    def __init__(self, reg_covar=1e-6,
                 weights_init=None, means_init=None, covariances_init=None,
                 random_state=None,
                 verbose=0):
        super(GaussianMixture, self).__init__(random_state=random_state)

        self.covariance_type = 'full'  # HARD CODE - DGME NEEDS FULL
        self.weights_init = weights_init
        self.means_init = means_init
        self.covariances_init = covariances_init

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.reg_covar < 0.:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, X.shape[1])

        if self.covariances_init is not None:
            self.covariances_init = _check_covariances(self.covariances_init,
                                                       self.covariance_type,
                                                       self.n_components,
                                                       X.shape[1])




    def _estimate_log_prob(self, X):
        estimate_log_prob_functions = {
            "full": _estimate_log_gaussian_prob_full,
            #"tied": _estimate_log_gaussian_prob_tied,
            #"diag": _estimate_log_gaussian_prob_diag,
            #"spherical": _estimate_log_gaussian_prob_spherical
        }
        return estimate_log_prob_functions[self.covariance_type](
            X, self.means_, self.covariances_)

    def _estimate_log_weights(self):
        normalized_weights = self.weights_ / np.sum(self.weights_)
        return np.log(normalized_weights)

    def _check_is_fitted(self):
        # since DGME is incremental, this isn't a good measure of whether
        # the model has been fit -- in fact, for incremental methods, there's
        # no clear concept of when the model has been fit.  Even after one
        # observation, you could say the model has been fit, but then it's only
        # been fit to that single observation...
        check_is_fitted(self, ['weights_', 'means_', 'covariances_'])

    def _get_parameters(self):
        return self.weights_, self.means_, self.covariances_

    def _set_parameters(self, params):
        self.weights_, self.means_, self.covariances_ = params

    def _n_parameters(self):
        """Return the number of free parameters in the model.

        Used in BIC and AIC computations.
        """
        ndim = self.means_.shape[1]
        n_components = self.means_.shape[0]
        if self.covariance_type == 'full':
            cov_params = n_components * ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = n_components * ndim
        elif self.covariance_type == 'tied':
            cov_params = ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = n_components
        mean_params = ndim * n_components

        return int(cov_params + mean_params + n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic: float
            The greater the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic: float
            The greater the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
