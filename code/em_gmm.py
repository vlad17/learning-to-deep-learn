import tensorflow as tf
import numpy as np
import sklearn
import random
import sys

# Algorithm copied pretty much directly from sklearn

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
    converged = False

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
                    converged = True
                    break
                prev_ll = ll
            m, s, a = sess.run((mus, sigmas, alphas))
    
    return ll, m, s, a, converged

# Given a set of partial observations xs each of dimension O < D for a fitted GMM model with 
# K cluster priors alpha, KxD means mus, and KxD diagonal covariances sigmas,
# returns the weighted sum of normals for the remaining D - O dimensions.
#
# Returns posterior_mus, posterior_sigmas, posterior_prior,
# of dimensions: Kx(D-O), Kx(D-O) for the posterior means and standard
# deviations and NxK for each x in xs representing the updated cluster
# weights conditioned on the partial observations given by xs.
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

class TFGMM(sklearn.base.BaseEstimator):
    """Gaussian Mixture Model implemented on top of TensorFlow.

    Only a diagonal model is representable.

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Initializes parameters such that every mixture component has zero
    mean and identity covariance.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    random_state: RandomState or an int seed (None by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold.  Defaults to 1e-3.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. the best results is kept

    verbose : boolean, optional (default false)
        Enable verbose output. 

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    covars_ : array
        Covariance parameters for each mixture component.
        the shape is (n_components, n_features), corresponding to the
        covariance matrix diagonal of each component in the diagonal
        GMM.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    """

    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1, verbose=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        self.verbose = verbose

        if n_init < 1:
            raise ValueError('GMM estimation requires at least one run')

        self.means_ = None

        # flag to indicate exit status of fit() method: converged (True) or
        # n_iter reached (False)
        self.converged_ = False

    def check_fitted(self):
        if self.means_ is None:
            raise Exception('GMM must be fitted')

    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        self.check_fitted()

        X = sklearn.utils.check_array(X)
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.ndim != 2:
            raise ValueError('X is not a 2-tensor (X.shape = {})'
                             .format(X.shape))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

        return estep(X, self.means_, self.covars_, self.weights_)

    def score(self, X, y=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob, _ = self.score_samples(X)
        return logprob

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,) component memberships
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gaussian
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities

    def fit_predict(self, X, y=None):
        """Fit and then predict labels for data.

        Warning: due to the final maximization step in the EM algorithm,
        with low iterations the prediction may not be 100% accurate

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,) component memberships
        """
        return self._fit(X, y).argmax(axis=1)

    def _fit(self, X, y=None, do_prediction=False):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the
        expectation-maximization (EM) algorithm. If you want to avoid
        this step, set the keyword argument init_params to the empty
        string '' when creating the GMM object. Likewise, if you would
        like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation.
        """

        # initialization step
        X = check_array(X, dtype=np.float64, ensure_min_samples=2,
                        estimator=self)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with {} components, but got only {} samples'
                .format(self.n_components, X.shape[0]))

        max_log_prob = -np.infty

        if self.verbose:
            print('Expectation-maximization algorithm started.')

        for init in range(self.n_init):
            if self.verbose:
                print('Initialization ' + str(init + 1))
                start_init_time = time()

            self.means_ = cluster.KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state).fit(X).cluster_centers_
            if self.verbose:
                print('\tMeans have been initialized.')

            ll, m, c, a, conv = fit_em(
                 X, self.means_, self.n_iter, self.tol,
                 min_covar=self.min_covar, verbose=self.verbose)
            
            if self.verbose and self.converged_:
                print('\t\tEM algorithm converged.')

            if ll > max_log_prob:
                max_log_prob = ll
                self.means_, self.covars_, self.weights, self.converged_ = (
                    m, c, a, conv)
                
                if self.verbose:
                    print('\tBetter parameters were found.')

            if self.verbose:
                print('\tInitialization ' + str(init + 1)
                      + ' took {0:.5f}s'.format(time() - start_init_time))

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        return self.predict_proba(X)

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the
        expectation-maximization (EM) algorithm. If you want to avoid
        this step, set the keyword argument init_params to the empty
        string '' when creating the GMM object. Likewise, if you would
        like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self._fit(X, y)
        return self

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.means_.shape[1]
        cov_params = self.n_components * ndim
        mean_params = ndim * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic: float (the lower the better)
        """
        return (-2 * self.score(X).sum() +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic: float (the lower the better)
        """
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()

