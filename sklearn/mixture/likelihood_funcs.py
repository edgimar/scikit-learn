"""
Likelihood functions and function-generators for use with DGME
"""
import numpy as np


def gen_likelihood_fn(q=0.9, N_inflection=31, N_q=60, epsilon=0.1,
                     min_to_max_ratio=0.1, sigma_abs_min=1e-7, denom=4):
    """Generate and return a default likelihood function.

    This function generates a default likelihood function for use with DGME.

    Parameters:

        N_inflection -- N value at which *alpha* (computed below) should be 0.5
        N_q -- N value at which *alpha* should have the value *q*
        epsilon -- how much to 'inflate' the true covariance by when computing the working cov.
        denom -- returned likelihood is the negative Mahalanobis distance divided by *denom*

        [ the following params are currently unused (associated w/ unused regularization below) ]
        min_to_max_ratio -- influences the "relative minimum" singular value limit
        sigma_abs_min -- "absolute minimum" singluar value permissible in sigma_reg
    """
    def likelihood_fn(m, i, obs):
        """Likelihood function which complements the *likelihood_threshold_fn* function
        """
        cov_max = m.max_cov_magnitude  # this value is defined when creating the model *m*

        # THIS FUNCTION IS BASED ON THE POPULATION COVARIANCE ESTIMATOR
        N = m.weights_[i]
        ndim = m.means_.shape[1]

        # compute the scaling factor alpha, which is a sigmoid function of N
        #   - when N << N_inflection, alpha ~= 0
        #   - when N >> N_inflection, alpha ~= 1
        a = (np.log(q) - np.log(1 - q)) / (N_q - N_inflection)
        alpha = 1.0 / (1 + np.exp(-a * (N - N_inflection)))

        # get current "actual" covariance estimate, based just on observed data
        #sigma_actual = (N - 1) / N * m.covariances_[i]  # a population covariance est.
        sigma_actual = m.covariances_[i]  # a sample covariance est.

        # in lieu of computing a regularized covariance below, we will set the
        # *sigma_reg* variable to use the unregularized ("actual") covariance
        sigma_reg = sigma_actual

        # COMPUTE REGULARIZED COVARIANCE
        #sigma_reg = sigma_actual + sigma_abs_min**2 * np.eye(ndim)

#        # GET LARGEST SING. VALUE of sigma_actual
#        u,s,v = np.linalg.svd(sigma_actual)
#        largest_sing_val = max(s)
#
#        # COMPUTE REGULARIZED COVARIANCE
#
#        # the min_to_max_ratio is itself scaled in the same way that alpha is scaled
#        # below (as a sigmoid function of N).
#        sigma_rel_min = min_to_max_ratio * (1 - alpha) * largest_sing_val
#        lower_limit = max(sigma_abs_min, sigma_rel_min)
#
#        # clamp small singular values
#        s[s < lower_limit] = lower_limit
#
#        # reconstruct matrix with clamped singular values
#        # NEED TO USE ONLY u or ONLY v -- OTHERWISE BECAUSE OF THE ARBITRARY SIGN-CHOICES USED
#        # WHEN COMPUTING BASIS VECTORS, USV MAY RESULT IN A NON-POSITIVE-DEFINITE MATRIX
#        sigma_reg = u.dot(np.diag(s)).dot(u.T)


        # COMPUTE "WORKING COVARIANCE"
        if N == 1:
            working_cov = cov_max * np.eye(ndim)
        else:
            working_cov = alpha * (1 + epsilon) * sigma_reg + \
                (1 - alpha) * cov_max * np.eye(ndim)

        # COMPUTE MAHALANOBIS DISTANCE USING "WORKING COV."

        # normalized log-density gives us -0.5 * mahalanobis distance
        negative_mahal_dist = 2 * m.estimate_log_gaussian_prob_full(obs, m.means_[i:(i + 1)],
                                                                    np.array([working_cov]),
                                                                    normalized=True)

        likelihood = negative_mahal_dist / denom

        return likelihood

    return likelihood_fn
