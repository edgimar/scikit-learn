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

from ..base import RegressorMixin
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


# TODO: store lower cholesky decomp of covariances, and only pass
#       these into this function, not the covariances.
def _estimate_log_gaussian_prob_full(X, means, covariances, normalized=False):
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
        # TODO: use cho_solve instead?  (it is faster; not sure about stability
        # differences)
        cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                         lower=True).T
        if normalized:
            log_prob[:, k] = - .5 * np.sum(np.square(cv_sol), axis=1)
        else:
            log_prob[:, k] = - .5 * (n_features * np.log(2. * np.pi) +
                                     cv_log_det +
                                     np.sum(np.square(cv_sol), axis=1))

    return log_prob




class DGMM(BaseMixture, RegressorMixin):
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
    # this class inherits from RegressorMixin, but this mixin has
    # a conflict with the predict() method as it's defined in the BaseMixture
    # class.  BaseMixture inherits from DensityMixin,
    # which sets the _estimator_type class-variable to "DensityEstimator", but
    # this attribute value isn't used anywhere, whereas the 'regressor' value
    # is checked in various places.  It's a bit of a mess how things are
    # currently structured with the sklearn inheritance hierarchy...
    # _estimator_type should probably be a list, not a string
    def __init__(self, reg_covar=1e-6,
                 weights_init=None, means_init=None, covariances_init=None,
                 random_state=None,
                 verbose=0,
                 init_cov_magnitude=None):
        super(DGMM, self).__init__(random_state=random_state)

        self.covariance_type = 'full'  # HARD CODE - DGME USES FULL
        self.reg_covar = reg_covar
        self.weights_init = weights_init
        # TODO: actually do something with means_init and covariances_init if
        # they are passed in (currently they aren't assigned to means_ and
        # covariances_).
        if (means_init is not None) or (covariances_init is not None):
            raise UserWarning("setting an initial set of means/covariances not yet supported.")
        self.means_init = means_init
        self.covariances_init = covariances_init

        if init_cov_magnitude is not None:
            # only set the attribute value if the constructor param isn't None
            self.init_cov_magnitude = init_cov_magnitude

    def __repr__(self):
        """Define string which represents this object when printed."""
        try:
            n_components = self.means_.shape[0]
        except AttributeError:
            n_components = 0

        return "DGMM: <%s components>" % str(n_components)

    def set_partition(self, input_component_indices,
                      output_component_indices=None):
        """Choose which features should be considered inputs and outputs."""
        self.input_indices = input_component_indices
        if output_component_indices is None:
            # construct output indices as complement to input indices
            try:
                n_features = self.means_.shape[1]
            except NameError:
                raise UserWarning("If the model hasn't yet been fit, "
                                  "set_partition() requires that both input "
                                  "and output indices be specified.")
            output_component_indices = list(range(n_features))
            for i in input_component_indices:
                output_component_indices.remove(i)

        self.output_indices = output_component_indices

    def fit(self, X, y=None):
        """Estimate model parameters.

        If *y* is provided, then after the model is fit, it will be partitioned
        accordingly so that the last N features will be considered to be output
        features (where N is the number of columns in *y*).

        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like, shape (n_observations, n_output_features)
            List of n_output_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        # this maybe isn't the best named function -- it's purpose here is not
        # to check whether the estimator has been fit, but rather whether the
        # init_cov_magnitude attribute has been set.
        check_is_fitted(self, ['init_cov_magnitude'],
                        msg="init_cov_magnitude hasn't been specified yet!")

        # initially, we will make DGMM.fit() simply create one Gaussian
        # per observation.
        # apparently the most efficient way to build an array rather than
        # appending to the array in a loop, is to use a list, and then
        # convert to an array (e.g. np.array(L))

        if y is not None:
            # reshape y into a column-vector if needed
            if y.ndim == 1:
                y = y.reshape((-1, 1))

            num_input_features = X.shape[1]
            num_output_features = y.shape[1]
            input_indices = np.arange(num_input_features)
            output_indices = num_input_features + np.arange(num_output_features)
            self.set_partition(input_indices, output_indices)

            # create new X where y augments the old X on the right hand side.
            X = np.hstack([X, y])

        # we will clear any previous parameters when fitting

        self.means_ = np.array(X)
        n_components = self.means_.shape[0]
        n_features = self.means_.shape[1]

        sigma_initial = np.eye(n_features) * self.init_cov_magnitude
        self.covariances_ = np.tile(sigma_initial, [n_components, 1, 1])
        self.weights_ = np.ones(n_components)

        return self

    def predict(self, input_vectors, min_weight=None):
        """
        If *min_weight* is set, only those components whose
        weights are equal to or higher than this value will be used for
        regression.  In other words, if *min_weight* = 2, then
        only components whose weights >= 2 will be used for the prediction.

        """
        # other mixture model classes use the predict() method to perform
        # classification, not regression, but we're making a break from that
        # here, and using it for regression.
        check_is_fitted(self, ['weights_', 'input_indices', 'output_indices'])
        return self.get_expected_ys(input_vectors, min_weight)

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




    def _estimate_log_prob(self, X, i=None):
        "If *i* is specified, only return the log-probability of the i-th component"
        if i is not None:
            log_probs = _estimate_log_gaussian_prob_full(X, self.means_[i:(i + 1)],
                                                         self.covariances_[i:(i + 1)])
        else:
            log_probs = _estimate_log_gaussian_prob_full(X, self.means_, self.covariances_)

        return log_probs

    @staticmethod
    def estimate_log_gaussian_prob_full(X, means, covariances, normalized=False):
        return _estimate_log_gaussian_prob_full(X, means, covariances, normalized)


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
                      self.input_indices,
                      self.output_indices)
        self.xy_covariances_ = self.covariances_[mesh]


        mesh = np.ix_(np.arange(self.covariances_.shape[0]),
                      self.output_indices,
                      self.output_indices)
        self.yy_covariances_ = self.covariances_[mesh]

        # now also compute and store the Cholesky decomposition for each
        # xx covariance matrix
        # TODO: only (re)compute this for components whose parameters have
        # changed

        cov_xx_inv_dot_cov_xy = []
        for i in range(self.covariances_.shape[0]):
            cov_xx = self.input_covariances_[i]
            cov_xy = self.xy_covariances_[i]
            cov_xx_inv_dot_cov_xy.append(linalg.cho_solve(linalg.cho_factor(cov_xx),
                                                          cov_xy))

        self.cov_xx_inv_dot_cov_xy_ = np.array(cov_xx_inv_dot_cov_xy)

    def _compute_conditional_covs(self):
        """Compute Cov[Y|x] for each component.

        Interestingly, the conditional covariance has no dependence on the
        specific value of the conditioning variable (i.e. X).  This means that
        it only needs to be computed once for a given GMM, and then can be used
        with several different *expected_ys_given_x* values that depend on
        different values of x.

        The conditional covariance is computed as:

            \Sigma_{YY} - \Sigma_{YX} \Sigma^{-1}_{XX} \Sigma_{XY}

        to which we can directly apply the values in *self.cov_xx_inv_dot_cov_xy_*.

        Returns a 3-dimensional array with shape (n_components, n_features, n_features)

        """
        # TODO: add to _compute_partitioned_means_and_covs ?
        # returns a 3D array : (num_means x num_input_features x num_input_features)
        B = self.cov_xx_inv_dot_cov_xy_

        # TODO: vectorize this (similar to how it was done in get_per_component_expected_ys())
        cond_covs = []
        for i in range(self.means_.shape[0]):
            cov_yy = self.yy_covariances_[i]
            cov_yx = self.xy_covariances_[i].T
            cov_xx_inv_dot_cov_xy = B[i]
            cov_y_given_x = cov_yy - cov_yx.dot(cov_xx_inv_dot_cov_xy)
            cond_covs.append(cov_y_given_x)

        cond_covs_array = np.array(cond_covs)
        self.y_given_x_covariances_ = cond_covs_array

    def _estimate_log_prob_input_pdf(self, X, min_weight=None):
        """
        X : array-like, shape (n_observations, n_input_features)
            It is assumed that the features in X are already ordered in a way
            that corresponds to the indices in *self.input_indices*

        Returns
        -------
        log_prob : array, shape (n_observations, n_components)
        """
        self._compute_partitioned_means_and_covs()

        if min_weight is None:
            means = self.input_means_
            input_covs = self.input_covariances_
        else:
            indices_to_use = self.weights_ >= min_weight
            means = self.input_means_[indices_to_use]
            input_covs = self.input_covariances_[indices_to_use]

        log_probs = _estimate_log_gaussian_prob_full(X, means, input_covs)

        return log_probs

    def get_per_component_expected_ys(self, X):
        """Compute E[Y|x] for each component, and each x in X.

        Returns a 3-dimensional array with shape:

                [(num components), (num xs), (num features per y)]

        """
        # returns a 3D array : (num_means x num_xs x num_input_features)
        deltaxs = X - self.input_means_[:, np.newaxis, :]
        B = self.cov_xx_inv_dot_cov_xy_
        expected_ys_given_x = self.output_means_[:, np.newaxis, :] + np.einsum('...ik,...kj', deltaxs, B)

        return expected_ys_given_x

    def get_conditional_component_weights(self, X):
        """Compute the overall E[Y|x] value for each x in X.

        X is a 2D array with one observation vector (x) per row.

        Returns:
            - conditional weights: a [(num mixture components),(num rows in X)] shaped array.

            (each column of this matrix represents a set of conditional component
            weights for a given x row-vector in X)

        """
        # first compute a matrix with row-vectors for each x value containing ln(w_i p_{X,i}) values
        ln_wi_pxi = self._estimate_weighted_log_prob_input_pdf(X, normalized_weights=False)

        # now make it so that instead of ln(w_i p_{X,i}) being in each row, it
        # contains normalized versions of these values where ln(sum_j(w_j
        # p_{X,j)) has been subtracted from each entry of the row
        ln_sum_wj_pxj = logsumexp(ln_wi_pxi, axis=1)
        ln_sum_wj_pxj.shape = (-1,1)  # one column, more than one row

        # result is a MxN array -- M x-values (rows), N components (cols)
        ln_normalized_wi_pxi = ln_wi_pxi - ln_sum_wj_pxj  # ln(k_j(x_i)) in each row i, column j

        component_weights = np.exp(ln_normalized_wi_pxi.T)

        return component_weights

    def get_expected_ys(self, X, min_weight=None):
        # TODO: change min_weight to a general list of component indices to use
        #       (or possibly some predicate function)
        """Compute the overall E[Y|x] value for each x in X.

        X is a 2D array with one observation vector (x) per row.

        Returns:
            - expected y values: a [(num rows in X),(num features per y)] shaped array.
        """
        component_weights = self.get_conditional_component_weights(X)

        # I don't know if there's another way to do this, but I am reshaping it
        # from (A,B) -> (A,B,1)
        component_weights.shape = tuple(list(component_weights.shape) + [1])

        expected_ys_given_x_all = self.get_per_component_expected_ys(X)

        if min_weight is None:
            expected_ys_given_x = expected_ys_given_x_all
            cweights = component_weights
        else:
            expected_ys_given_x = expected_ys_given_x_all[self.weights_ >= min_weight]
            cweights = component_weights[self.weights_ >= min_weight]

        weighted_expected_ys = expected_ys_given_x * cweights

        # now I want to do a sum of all first-rows, all second-rows, etc.
        # EACH ROW OF THIS IS THE EXPECTED Y CORRESPONDING TO ONE OF THE PROVIDED X VALUES
        overall_expected_ys = np.sum(weighted_expected_ys, axis=0)

        return overall_expected_ys

    def get_pdf(self, X, normalized=True):
        # TODO: rename *normalized* param to something else -- it currently means
        #       'normalized weights', though one might mistakenly assume that it
        #       means 'normalized pdf'.
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


    # score() code taken directly from RegressorMixin -- included here to
    # prevent inheriting the wrong score() method
    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the regression
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        from ..metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


    # score_basemixture() taken from BaseMixture class -- was originally just
    # called score()
    def score_basemixture(self, X, y=None):
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

    def _estimate_weighted_log_prob_input_pdf(self, X, normalized_weights=True,
                                              min_weight=None):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_observations, n_components)
        """
        return self._estimate_log_prob_input_pdf(X, min_weight) + \
            self._estimate_log_weights(normalized_weights, min_weight)

    def _estimate_log_weights(self, normalized_weights=True,
                            min_weight=None):
        if min_weight is None:
            w = self.weights_
        else:
            w = self.weights_[self.weights_ >= min_weight]

        if normalized_weights:
            weights = w / np.sum(w)
        else:
            # use unnormalized weights
            weights = w

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
        return (-2 * self.score_basemixture(X) * X.shape[0] +
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
        return -2 * self.score_basemixture(X) * X.shape[0] + 2 * self._n_parameters()


class DGMMOnline(DGMM):
    """Dynamic Gaussian Mixture Model - Online Learning

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
    def __init__(self, max_cov_magnitude, reg_covar=1e-6,
                 weights_init=None, means_init=None, covariances_init=None,
                 random_state=None,
                 verbose=0,
                 init_cov_magnitude=None):

        super(DGMMOnline, self).__init__(reg_covar, weights_init, means_init,
                covariances_init, random_state, verbose, init_cov_magnitude)

        self.max_cov_magnitude = max_cov_magnitude
        self.timelapse_enabled = False
        # BEGIN: IN SUPERCLASS
        #self.covariance_type = 'full'  # HARD CODE - DGME NEEDS FULL
        #self.weights_init = weights_init
        #self.means_init = means_init
        #self.covariances_init = covariances_init

        #if init_cov_magnitude is not None:
        #    # only set the attribute value if the constructor param isn't None
        #    self.init_cov_magnitude = init_cov_magnitude
        # END: IN SUPERCLASS

    def set_timelapse(self, timelapse_enabled):
        self.timelapse_enabled = timelapse_enabled

    def set_penalized_likelihood_fn(self, fn):
        # *fn* is a function accepting the arguments:
        #   (DGMM object, i, observation)
        # - it will return a likelihood of *observation* given the *i*th
        #   component in the model.
        self.penalized_likelihood = fn

    def set_penalized_likelihood_threshold_fn(self, fn):
        # *fn* is a function accepting the arguments:
        #   (n_features, max_cov_magnitude)
        # - it returns a threshold value that is compared against values from
        #   self.penalized_likelilhood function, to determine if an observation
        #   should be merged or added.
        self.penalized_likelihood_threshold_fn = fn

    def fit(self, X, y=None):
        """Estimate model parameters, clearing previous parameters.

        If *y* is provided, then after the model is fit, it will be partitioned
        accordingly so that the last N features will be considered to be output
        features (where N is the number of columns in *y*).

        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like, shape (n_observations, n_output_features)
            List of n_output_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        # clear the model parameters, then run partial_fit on the data
        try:
            del(self.means_)
            del(self.covariances_)
            del(self.weights_)
        except AttributeError:
            # they don't currently exist, so no need to remove them
            pass

        return self.partial_fit(X, y)


    #  this method is somehow part of the sklearn API -- sees
    #  http://scikit-learn.org/stable/modules/scaling_strategies.html where it
    #  is mentioned.  It is basically an incremental fit method.
    def partial_fit(self, X, y=None):
        """Update model parameters using data X (and optional targets y).

        If *y* is provided, then after the model is fit, it will be partitioned
        accordingly so that the last N features will be considered to be output
        features (where N is the number of columns in *y*).

        While there may be better ways to merge batches of data, for now, we
        will just integrate one observation at a time from the batch, until all
        have been integrated into the model.

        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like, shape (n_observations, n_output_features)
            List of n_output_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        # this maybe isn't the best named function -- it's purpose here is not
        # to check whether the estimator has been fit, but rather whether the
        # init_cov_magnitude attribute has been set.
        check_is_fitted(self, ['init_cov_magnitude', 'max_cov_magnitude',
                    'penalized_likelihood_threshold_fn', 'penalized_likelihood'],
                        msg="unspecified hyperparameters or likelihood functions")

        if y is not None:
            # reshape y into a column-vector if needed
            if y.ndim == 1:
                y = y.reshape((-1, 1))

            num_input_features = X.shape[1]
            num_output_features = y.shape[1]
            input_indices = np.arange(num_input_features)
            output_indices = num_input_features + np.arange(num_output_features)
            self.set_partition(input_indices, output_indices)

            # create new X where y augments the old X on the right hand side.
            X = np.hstack([X, y])

        if self.timelapse_enabled:
            # initialize temporary model-holding-spot
            self.models = []

        # now update the parameters using online DGME
        for x in X:
            self._update_single(x)
            if self.timelapse_enabled:
                self.models.append((self.means_.copy(),
                self.covariances_.copy(), self.weights_.copy()))

        return self

    def _set_component_params(self, i, mu, sigma, weight):
        "Set the parameters of the *i*th component"
        self.means_[i] = mu
        self.covariances_[i] = sigma
        self.weights_[i] = weight

    def _do_merge_to_gaussian(self, x, i):
        """merge *x* into component *i*, replacing the component's parameters.

        The equations used for this update (in LaTeX format), taken from
        "Dynamic Motion Modelling for Legged Robots" (Edgington, 2009) are:

        w_{new} &= w_{old} + 1

        \vec{\mu}_{new} &= \frac{w_{old}}{w_{new}} \vec{\mu}_{old}
                                                + \frac{1}{w_{new}} \vec{x}
                       AND

        \Sigma_{new} &= \frac{w_{old}-1}{w_{old}} \Sigma_{old}
                      + \vec{\mu}_{old} \vec{\mu}_{old}^T
              & \quad + \frac{1}{w_{old}} \vec{x} \vec{x}^T
                      - \frac{w_{new}}{w_{old}} \vec{\mu}_{new} \vec{\mu}_{new}^T

        Note: all of the vectors in the above equations are column-vectors,
              whereas *x* and *mu* are row-vectors in this method.

        """
        # get old parameters
        mu = self.means_[i]
        cov = self.covariances_[i]
        w = self.weights_[i]

        # compute updated parameters
        w2 = w + 1
        mu2 = ((w * mu) + x) / w2
        cov2 = np.outer(mu,mu) + \
            ((w - 1) * cov + np.outer(x, x) - w2 * np.outer(mu2, mu2)) / w

        # add regularization to cov2 -- *reg_covar* is a float that is added to
        # the diagonal of cov2
        cov2.flat[::self.means_.shape[1] + 1] += self.reg_covar

        # set the new parameters
        self._set_component_params(i, mu2, cov2, w2)

    def _add_new_gaussian(self, x, first=False):
        "create a new Gaussian for the observation *x*"
        sigma = np.eye(x.shape[1]) * self.init_cov_magnitude

        if first:
            self.means_ = x
            self.covariances_ = np.array([sigma])
            self.weights_ = np.array([1])
        else:
            # append mu to end of means_ array
            self.means_ = np.vstack([self.means_, x])

            # append sigma to end of covariances_ array (sigma must be a
            # 3-dimensional array to stack below a 3D array)
            self.covariances_ = np.vstack([self.covariances_,
                                           sigma[np.newaxis, :, :]])

            # initial weight is always 1
            self.weights_ = np.concatenate([self.weights_, [1]])


    def _update_single(self, x):
        """
        Update the mixture model with a single observation vector.

        The observation vector *x* must be a numpy array or row-vector-matrix.

        """
        x = np.array(x)
        # reshape x into a row-vector
        x = x.reshape((1, -1))

        # check for valid shape/size of x
        try:
            n_features = self.means_.shape[1]
            _check_shape(x, (1, n_features), 'x')
        except AttributeError:
            # self.means_ doesn't exist yet
            self._add_new_gaussian(x, first=True)
            n_features = self.means_.shape[1]
            _check_shape(x, (1, n_features), 'x')

            # compute likelihood threshold (one-time only)
            self.likelihood_threshold = \
                self.penalized_likelihood_threshold_fn(n_features,
                                            self.max_cov_magnitude)

            return self

        ## [temp-removed] filter out (remove) all dormant gaussians from consideration

        ###### integrate observation into model #####

        # build list of (L(x; i, means_, covariances_, weights_), i) tuples

        likelihoods = []
        n_components = self.means_.shape[0]
        for i in range(n_components):
            likelihoods.append((self.penalized_likelihood(self, i, x), i))

        max_penalized_likelihood, i = max(likelihoods)


        # decide whether to merge the observation into one of the model's
        # existing Gaussians, or to create a new Gaussian in the model

        # merge decision criterion (boolean variable)
        do_merge = max_penalized_likelihood >= self.likelihood_threshold

        if do_merge:
            self._do_merge_to_gaussian(x, i)
        else:
            self._add_new_gaussian(x)

