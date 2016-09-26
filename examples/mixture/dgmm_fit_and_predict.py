"""
=========================================
Use DGME to fit GMM and Predict using GMM
=========================================

This example fits a GMM to a small number of observations using Dynamic
Gaussian Mixture Estimation (DGME), and then makes predictions based on
partial-observations using Gaussian Mixture Regression (GMR).

"""
import numpy as np

from sklearn import mixture

np.random.seed(2345)

# 6 features, 8 observations ; each feature distributed uniformly over [0,1)
# (each observation is a row-vector)
data = np.random.rand(8,6)

# create and set-up the model
m = mixture.DGMMOnline(init_cov_magnitude=0.1, max_cov_magnitude=0.3)
m.set_penalized_likelihood_fn(mixture.likelihood_funcs.gen_likelihood_fn())
m.set_penalized_likelihood_threshold_fn(lambda n_feat, max_cov: -1.0)

# use DGME to learn the model parameters from the data
m.fit(data)
print("GMM fit using DGME to {} random observations taken from a uniform"
      " distribution over [0,1).\nResulting model contains {} mixture components."
      "".format(data.shape[0], m.means_.shape[0]))

# specify that the x-subvector consists of elements 0 to 3 of each data vector
m.set_partition([0,1,2,3])

# take x subvectors from the first two rows of data
xs = data[0:2, 0:4]

print('\nPredicting observation elements 4 and 5 (the output sub-vector) given'
      '\n the following sets of values for elements 0,1,2,3 (the input sub-vector):')
print(xs)

expected_ys = m.get_expected_ys(xs)
actual_ys = data[0:2, 4:]
print('\nPredicted y-vectors:')
print(expected_ys)

print('\nActual y-vectors:')
print(actual_ys)

print('\nFractional Error:')
print((expected_ys - actual_ys) / actual_ys)
