import numpy as np
from numpy import pi, sqrt, exp,log
from scipy.stats import multivariate_normal

def GAUSSIAN(male_mean,male_var,male_prior,female_mean,female_var,female_prior,test_X):
    predict = np.zeros(test_X.shape[0])
    for i in range(test_X.shape[0]):
        p_male = ((1 / sqrt(2 * pi * male_var) * exp(-(test_X[i] - male_mean) ** 2 / (2 * male_var))).prod(
            axis=0)) * male_prior
        p_female = ((1 / sqrt(2 * pi * female_var) * exp(-(test_X[i] - female_mean) ** 2 / (2 * female_var))).prod(
            axis=0)) * female_prior
        prob = p_male + p_female
        p_male /= prob
        p_female /= prob
        if (p_male >= p_female*10):
            predict[i] = 0
        else:
            predict[i] = 1
    return predict


def N_GAUSSIAN(male_mean,male_cov,male_prior,female_mean,female_cov,female_prior,test_X):
    predict = np.zeros(test_X.shape[0])

    for i in range(test_X.shape[0]):
        p_male = multivariate_normal.logpdf(test_X[i], mean=male_mean, cov=male_cov) + log(male_prior)
        p_female = multivariate_normal.logpdf(test_X[i], mean=female_mean, cov=female_cov) + log(female_prior)
        if (p_male >= p_female):
            predict[i] = 0
        else:
            predict[i] = 1
    return predict