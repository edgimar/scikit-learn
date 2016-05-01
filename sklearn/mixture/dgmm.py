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

from .base_light import BaseMixture, _check_shape, _check_X
from ..externals.six.moves import zip
from ..utils import check_array
from ..utils.validation import check_is_fitted
from ..utils.extmath import logsumexp


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
    X : array-like, shape (n_observations, n_features)

    means : array-like, shape (n_components, n_features)

    covariances : array-like, shape (n_components, n_features, n_features)

    Returns
    -------
    log_prob : array, shape (n_observations, n_components)
    """
    n_observations, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_observations, n_components))
    for k, (mu, cov) in enumerate(zip(means, covariances)):
        try:
            cov_chol = linalg.cholesky(cov, lower=True)
        except linalg.LinAlgError:
            raise ValueError("The algorithm has diverged because of too "
                             "few observations per component. "
                             "Try to decrease the number of components, or "
                             "increase reg_covar.")
        cv_log_det = 2. * np.sum(np.log(np.diagonal(cov_chol)))
        cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                         lower=True).T
        log_prob[:, k] = - .5 * (n_features * np.log(2. * np.pi) +
                                 cv_log_det +
                                 np.sum(np.square(cv_sol), axis=1))
    return log_prob


class DGMM(BaseMixture):
    """Dynamic Gaussian Mixture Model

    Representation of a Gaussian mixture model probability distribution.


    Parameters
    ----------
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
        super(DGMM, self).__init__(random_state=random_state)

        self.covariance_type = 'full'  # HARD CODE - DGME NEEDS FULL
        self.weights_init = weights_init
        self.means_init = means_init
        self.covariances_init = covariances_init

    def __repr__(self):
        """Define string which represents this object when printed."""
        n_components = self.means_.shape[0]
        return "DGMM: <%s components>" % str(n_components)

    def set_partition(self, input_component_indices,
                      output_component_indices=None):
        """Choose which features should be considered inputs and outputs."""
        self.input_indices = input_component_indices
        if output_component_indices is None:
            # construct output indices as complement to input indices
            n_features = self.means_.shape[1]
            output_component_indices = list(range(n_features))
            for i in input_component_indices:
                output_component_indices.remove(i)

        self.output_indices = output_component_indices

    # TODO: is there a reason y is needed by the sklearn API?  If not,
    # then remove it since it doesn't make sense here.  The model
    # should always be fit to "full" observation vectors.
    def fit(self, X, y=None):
        """Estimate model parameters.

        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        # initially, we will make DGMM.fit() simply create one Gaussian
        # per observation.
        # apparently the most efficient way to build an array rather than
        # appending to the array in a loop, is to use a list, and then
        # convert to an array (e.g. np.array(L))

        # we will clear any previous parameters when fitting

        self.means_ = np.array(X)
        n_components = self.means_.shape[0]
        n_features = self.means_.shape[1]

        # TODO: replace 0.01 with self.init_cov_magnitude
        sigma_initial = np.eye(n_features) * 0.01
        self.covariances_ = np.tile(sigma_initial, [n_components, 1, 1])
        self.weights_ = np.ones(n_components)

        return self

    def predict(self, input_vector):
        # sklearn typically uses methods with this name to perform
        # classification, not regression.
        check_is_fitted(self, ['weights_', 'input_indices', 'output_indices'])
        pass

    def predict_y(self, input_vectors):
        """Get expected value of outputs features given an input feature vector

        The overall procedure required here is as follows:

        1. Compute 3D array A with M 'rows' that look like:

                [ ln(E_{1}[Y|X=x] ln(E_{1}[Y|X=x] ... ln(E_{N}[Y|X=x] ]

            - M = number of rows in *input_vectors*
            - N = number of mixture components
            - each entry of this MxN 'matrix' is a P dimensional vector, where
              P is the number of output features.

        2. Compute a 2D array B with M rows that look like:

                [ ln(w_1 p_{X,1}(X=x) ln(w_2 p_{X,2}(X=x) ... ln(w_N p_{X,N}(X=x) ]

            - M = number of rows in *input_vectors*
            - N = number of mixture components
            - w_i = the unnormalized weight of the i^{th} mixture component
            - p_{X,i} = the marginal pdf of the input-features for mixture component i

        3. Compute a 1xM array C where each row (element) of the array is:

                [ ln( \sum_i w_i P_{X,i}(X=x) ) ]

            - this is computed from the step 2 array with the logsumexp() function

        4. Compute an MxN array D in which each row of the step 2 array has had
           the corresponding row of the step 3 array subtracted from it.
            - the elements in each row of this array will be ln(k_i(x)), where
              x is the input vector in *input_vectors* at the same row number.
              k_i(x) is the "per-mixture-component weight" at x, and this value
              is multiplied with E_{i}[Y|X=x] in order to produce one of the
              summands in the sum which results in the overall E[Y|X=x] value
              for a given input.

        5. Compute the array F as the element-wise sum of A + D
            - this will result in a 3D array, where, for example,
                F_{11} = A_{11} + B_{11}  (F_{11} is a vector)
            - A_{11} is a P-element vector (see step 1 above)
            - B_{11} is a scalar, which is added to every element of A_{11}

        6. Compute a matrix G, where the first two rows looks like:

                [ logsumexp_{i} ( F_{1i}) ]
                [ logsumexp_{i} ( F_{2i}) ]

            - i goes from 1 to N (# of components)
            - the first row is a vector whose values are equivalent to
              ln(m(x_1)), where m(x_1) is E[Y | X = x_1]

        7. Return np.exp(G)

        """
        check_is_fitted(self, ['weights_', 'input_indices', 'output_indices'])
        pass

    def add_gaussian(self, mu, sigma, weight=1):
        pass

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
        log_probs = _estimate_log_gaussian_prob_full(X, self.means_, self.covariances_)

        return log_probs

    def _compute_partitioned_means_and_covs(self):
        """Use the chosen input indices to get input means and covariances.

        TODO:  only (re)compute these if a dirty flag is set -- this will be
               set any time the model has been updated / modified in some way.
        """
        check_is_fitted(self, ['input_indices', 'output_indices'])
        # get "input" version of means and covariances, based on self.input_indices
        #self.means_ : array-like, shape (n_components, n_features)
        self.input_means_ = self.means_[:, self.input_indices]
        self.output_means_ = self.means_[:, self.output_indices]

        # compute cartesian product of indices (an array  of coordinate tuples,
        # each which identify a single element within the 3D covariances_ array.
        mesh = np.ix_(np.arange(self.covariances_.shape[0]),
                      self.input_indices,
                      self.input_indices)

        # self.covariances_ : array-like, shape (n_components, n_features, n_features)
        self.input_covariances_ = self.covariances_[mesh]

        mesh = np.ix_(np.arange(self.covariances_.shape[0]),
                      self.output_indices,
                      self.input_indices)
        self.yx_covariances_ = self.covariances_[mesh]

        # now also compute and store the Cholesky decomposition for each
        # xx covariance matrix
        # TODO: only (re)compute this for components whose parameters have
        # changed

        xx_choleskys = []
        for i in range(self.covariances_.shape[0]):
            cov = self.input_covariances_[i]
            # append matrices that can be used w/ scipy.linalg.cho_solve()
            xx_choleskys.append(linalg.cho_factor(cov))

        self.input_choleskys_ = np.array(xx_choleskys)

    def _estimate_log_prob_input_pdf(self, X):
        """
        X : array-like, shape (n_observations, n_input_features)
            It is assumed that the features in X are already ordered in a way
            that corresponds to the indices in *self.input_indices*

        Returns
        -------
        log_prob : array, shape (n_observations, n_components)
        """
        self._compute_partitioned_means_and_covs()

        log_probs = _estimate_log_gaussian_prob_full(X, self.input_means_,
                                                     self.input_covariances_)

        return log_probs


    def get_pdf(self, X, normalized=True):
        # TODO: rename "normalized" to something else, because it doesn't mean
        #       'normalized pdf', but 'normalized weights', which is confusing.
        """Return the total probability density of the mixture at *x*.

        This is calculated as p(x) = \sum_i [w_i * p_i(x)] where
        p_i is the probability of a single Gaussian component, and w_i is its
        associated normalized weight.

        If *normalized* is False, then unnormalized weights are used.

        Return None if the mixture model does not yet contain any Gaussians.

        """
        n_components = self.means_.shape[0]
        if n_components == 0:
            return None

        # if X has 5 rows (i.e. 5 observations), then log_p_X will be a
        # 5-element column vector containing log(p(x_i)) for each row i of X
        log_p_X = self.score_samples(X, normalized_weights=normalized)

        return np.exp(log_p_X)

    def get_input_pdf(self, X, normalized=True):
        """Return the total probability density of the mixture at *x*.

        This is calculated as p(x) = \sum_i [w_i * p_i(x)] where
        p_i is the probability of a single Gaussian component, and w_i is its
        associated normalized weight.

        If *normalized* is False, then unnormalized weights are used.

        Return None if the mixture model does not yet contain any Gaussians.

        """
        n_components = self.means_.shape[0]
        if n_components == 0:
            return None

        # if X has 5 rows (i.e. 5 observations), then log_p_X will be a
        # 5-element column vector containing log(p(x_i)) for each row i of X
        log_p_X = self.score_samples_input_pdf(X, normalized_weights=normalized)

        return np.exp(log_p_X)

    def score_samples_input_pdf(self, X, normalized_weights=True):
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_observations,)
            Log probabilities of each data point in X.
        """
        self._check_is_fitted()
        X = _check_X(X, None, len(self.input_indices))

        weighted_log_probs = self._estimate_weighted_log_prob_input_pdf(X, normalized_weights)
        return logsumexp(weighted_log_probs, axis=1)

    def _estimate_weighted_log_prob_input_pdf(self, X, normalized_weights=True):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_observations, n_components)
        """
        return self._estimate_log_prob_input_pdf(X) + self._estimate_log_weights(normalized_weights)

    def _estimate_log_weights(self, normalized_weights=True):
        if normalized_weights:
            weights = self.weights_ / np.sum(self.weights_)
        else:
            # use unnormalized weights
            weights = self.weights_

        return np.log(weights)

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
        X : array of shape (n_observations, n_dimensions)

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
        X : array of shape(n_observations, n_dimensions)

        Returns
        -------
        aic: float
            The greater the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
