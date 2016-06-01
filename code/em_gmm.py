import tensorflow as tf
import numpy as np
import sklearn
import random
import sys

# Algorithm copied pretty much directly from sklearn
# TODO: make sklearn-like interface with extra MLE method.
# (To make MLE faster, make an aggr TF method that computes them in
# parallel - outside this file, using a queuerunner).

#class TFGMM(sklearn.base.BaseEstimator):
# TODO use the parameter dtype.
# TODO with tf.Graph().as_default() statements.

# Returns a TensorFlow scalar with the size of the i-th dimension for
# the parameter tensor x.
def tf_get_shape(x, i):
    return tf.squeeze(tf.slice(tf.shape(x), [i], [1])) 

def tf_nrows(x):
    return tf_get_shape(x, 0)

def tf_ncols(x):
    return tf_get_shape(x, 1)

# Simultaneous K-cluster likelihood computation.
# X is NxD, mus is KxD, sigmas is KxD
# Output is KxN likelihoods for each sample in each cluster.
def tf_log_normals(X, mus, sigmas):
    # p(X) = sqrt(a * b * c)
    # a = (2 pi)^(-p)
    # b = det(sigma)^(-1)
    # c = exp(-(x - mu)^T sigma^(-1) (x - mu)), expanded for numerical stability
    #
    # Below we make simplifications since sigma is diag
    
    D = tf_ncols(mus)
    XT = tf.transpose(X) # pxN
    invsig = tf.inv(sigmas)

    dtype = XT.dtype
    
    loga = -tf.cast(D, dtype) \
           * tf.log(tf.constant(2 * np.pi, dtype)) # scalar
    logb = tf.reduce_sum(tf.log(invsig), 1, keep_dims=True) # Kx1
    logc =  \
        - tf.reduce_sum(invsig * tf.square(mus), 1, keep_dims=True) \
        + 2 * tf.matmul(invsig * mus, XT) \
        - tf.matmul(invsig, tf.square(XT)) # KxN
    
    return 0.5 * (loga + logb + logc)

# Stably log-sum-exps likelihood along rows.
# Reduces KxN tensor L to 1xN tensor
def tf_log_sum_exp(L):
    maxs = tf.reduce_max(L, 0, keep_dims=True) # 1xN
    return tf.log(tf.reduce_sum(tf.exp(L - maxs), 0, keep_dims=True)) + maxs

# X is NxD, mus is KxD, sigmas KxD, alphas is K
# output is KxN log likelihoods.
def tf_log_likelihood(X, mus, sigmas, alphas):
    alphas = tf.expand_dims(alphas, 1) # Kx1
    return tf_log_normals(X, mus, sigmas) + tf.log(alphas) # KxN

# X is NxD, mus is KxD, sigmas KxD, alphas is K
# output is 1xN log probability for each sample, KxN responsibilities
def estep(X, mus, sigmas, alphas):
    log_likelihoods = tf_log_likelihood(X, mus, sigmas, alphas)
    sample_log_prob = tf_log_sum_exp(log_likelihoods) # 1xN
    return sample_log_prob, tf.exp(log_likelihoods - sample_log_prob)

EPS = np.finfo(float).eps
MIN_COVAR_DEFAULT = EPS

# X is NxD, resp is KxN (and normalized along axis 0)
# Returns maximize step means, covariance, and cluster priors,
# which have dimension KxD, KxD, and K, respectively
def mstep(X, resp, min_covar=MIN_COVAR_DEFAULT):
    weights = tf.reduce_sum(resp, 1) # K
    invweights = tf.expand_dims(tf.inv(weights + 10 * EPS), 1) # Kx1
    alphas = EPS + weights / (tf.reduce_sum(weights) + 10 * EPS) # K
    weighted_cluster_sum = tf.matmul(resp, X) # KxD 
    mus = weighted_cluster_sum * invweights
    avg_X2 = tf.matmul(resp, tf.square(X)) * invweights
    avg_mu2 = tf.square(mus)
    avg_X_mu = mus * weighted_cluster_sum * invweights
    sigmas = avg_X2 - 2 * avg_X_mu + avg_mu2 + min_covar
    return mus, sigmas, alphas

# Similar pattern to
# https://gist.github.com/narphorium/d06b7ed234287e319f18

# Runs up to max_steps EM iterations, stopping earlier if log likelihood improves
# less than tol.
# X should be an NxD data matrix, initial_mus should be KxD
# max_steps should be an int, tol should be a float.
def fit_em(X, initial_mus, max_steps, tol, min_covar=MIN_COVAR_DEFAULT, verbose=False):
    N, D = X.shape
    K, Dmu = initial_mus.shape
    assert D == Dmu
        
    mus0 = initial_mus
    sigmas0 = np.tile(np.var(X, axis=0), (K, 1))
    alphas0 = np.ones(K) / K

    with tf.Graph().as_default():
        X = tf.constant(X)
        
        mus, sigmas, alphas = (tf.Variable(x, dtype=X.dtype)
                               for x in [mus0, sigmas0, alphas0])
        
        all_ll, resp = estep(X, mus, sigmas, alphas)
        cmus, csigmas, calphas = mstep(X, resp, min_covar=min_covar)
        update_mus_step = tf.assign(mus, cmus)
        update_sigmas_step = tf.assign(sigmas, csigmas)
        update_alphas_step = tf.assign(alphas, calphas)     
        
        init_op = tf.initialize_all_variables()
        ll = prev_ll = -np.inf
        
        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(max_steps):
                ll = sess.run(tf.reduce_mean(all_ll))
                sess.run((update_mus_step, update_sigmas_step, update_alphas_step))
                if verbose: print('EM iteration', i, 'log likelihood', ll)
                if abs(ll - prev_ll) < tol:
                    break
                prev_ll = ll
            m, s, a = sess.run((mus, sigmas, alphas))
    
    return ll, m, s, a

# Given a set of partial observations xs each of dimension O < D for a fitted GMM model with 
# K cluster priors alpha, KxD means mus, and KxD diagonal covariances sigmas,
# returns the weighted sum of normals for the remaining D - O dimensions.
#
# Returns posterior_mus, posterior_sigmas, posterior_prior,
# of dimensions:
# Kx(D-O), Kx(D-O), NxK, respectively (each mu, sigma is the same for all posteriors).
# NxK, NxKxD, NxKxD, respectively, for each x in xs, total of N.
def marginal_posterior(xs, mus, sigmas, alphas):
    # https://gbhqed.wordpress.com/2010/02/21/conditional-and-marginal-distributions-of-a-multivariate-gaussian/
    # diagonal case is easy:
    # https://en.wikipedia.org/wiki/Schur_complement#Applications_to_probability_theory_and_statistics
    O = xs.shape[1]
    D = mus.shape[1]
    with tf.Graph().as_default():
        dtype = tf.as_dtype(xs.dtype)
        observed_mus, observed_sigmas = (tf.constant(a, dtype=dtype)
                                         for a in (mus[:,0:O], sigmas[:, 0:O]))
        ll = tf_log_likelihood(xs, observed_mus, observed_sigmas, alphas) # KxN
        norm = tf_log_sum_exp(ll) # 1xN
        with tf.Session() as sess:
            ll, norm = sess.run((ll, norm))
    return mus[:, O:D], sigmas[:, O:D], np.transpose(ll - norm)

# A "sparser" estimate which just uses the most likely cluster's mean as the estimate.
def argmax_exp(mus, sigmas, alphas):
    i = np.argmax(alphas)
    return mus[i]

# The originating notebook, from final-proj-cos-424, had a method for
# gradient descent on the negative log likelihood of the posterior GMM model.
# However, the absolute MLE is very close to one of the normal's expectations,
# because of the exponential decay of the normal pdf past the inflection point,
# which, for high dimensions, occurs very quickly. As such, it does not make
# sense to have a complicated, slow algorithm when the most-likely cluster
# mean is an estimat that's just as good.
