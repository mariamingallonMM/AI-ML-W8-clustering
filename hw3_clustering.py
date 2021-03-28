"""
This code implements a K-means and EM Gaussian mixture models per week 8 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.7 for running on Vocareum

Execute as follows:
$ python3 hw3_clustering.py X.csv
"""

# builtin modules
import sys
import os
import math
from random import randrange
import functools
import operator
import requests
import psutil

# 3rd party modules
import numpy as np
import pandas as pd
import scipy as sp
from scipy.cluster.vq import kmeans2
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy import stats


def KMeans(data, k, iterations, **kwargs):
    """
    It uses the scipy.klearn2 algorithm to classify a set of observations into k clusters using the k-means algorithm (scipy.kmeans2). It attempts to minimize the Euclidean distance between observations and centroids. Note the following methods for initialization are available:
        - ‘random’: generate k centroids from a Gaussian with mean and variance estimated from the data.
        - ‘points’: choose k observations (rows) at random from data for the initial centroids.
        - ‘++’: choose k observations accordingly to the kmeans++ method (careful seeding)
        - ‘matrix’: interpret the k parameter as a k by M (or length k array for 1-D data) array of initial centroids.

    It is recommended the kmeans algorithm is initialized by randomly selecting 5 data points (minit = "points", with k = 5). Also, note we use the identity matrix for each cluster's covariance matrix for the initialization. We also try initializing data points without replacement: for example, using np.random.choice and set replace = False.
    ------------
    Parameters:

    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    - k: number of clusters to consider, 5 clusters by default
    - iterations: number of iterations to consider, 10 iterations by default
    ------------
    Returns:

    - centroids, which are the means for each cluster {μ1,…,μK}
    - labels, are the corresponding assignments of each data point {c1,…,cn}, where each ci ∈ {1,…,K} and ci indicates which of the K clusters the observation xi belongs to.
    - writes the centroids of the clusters associated with each iteration to a csv file, one file per iteration; pass on a path if different from the default being "current working directory" + "outputs"
    """

    #data = data.to_numpy() # Convert dataframe to np array
    centroids_list = []
    labels_list = []

    for i in range(iterations):
        centroids, label = kmeans2(data, k, iter = i+1, minit='points')
        centroids_list.append(centroids)
        labels_list.append(label)
        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        if 'path' in kwargs:
            path = kwargs['path']
            filepath = os.path.join(path, filename)
            np.savetxt(filepath, centroids, delimiter=",")
        else:
            path = os.getcwd() # os.path.join(os.getcwd(), "outputs")
            filepath = os.path.join(path, filename)
            np.savetxt(filepath, centroids, delimiter=",")
    
    return centroids_list, labels_list

def calculate_mean_covariance(data, centroids, labels):
    """
    Calculates means and covariance of different clusters from k-means prediction. This helps us calculate values for our initial parameters. It takes in our data as well as our predictions from k-means and calculates the weights, means and covariance matrices of each cluster. Note that we shall use the results from each of the iterations of k-means algorithm.

    Note that 'counter' is equivalent to 'cluster_label' provided the clusters are of int type and ranging from '0' to 'k'. We could have simplified counter = cluster_label but we chose not to do so to allow for cases in which the lable is not an integer.
    ------------
    Parameters:

    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    - centroids: centroids from KMeans method
    - labels: cluster labels from KMeans method, note it includes all the iterations
    - k: number of clusters (also number of Gaussians)

    -------------
    Returns:

    A tuple containing:
        
    - initial_pi: initial array of pik (πk = nk / n) values for each cluster k to input in the E-step of EM algorithm (shape: k,)
    - initial_mean: initial array of mu (μk, mean) values for each cluster kto input in the E-step of EM algorithm (shape: k, d)
    - initial_sigma: initial array of covariance (Σk, sigma) values for each cluster k to input in the E-step of EM algorithm (shape: k, d*d)
        
    """
    # initialize the output ndarrays with zeros, for filling up in loops below
    d = data.shape[1]
    k = len(centroids[0]) # to take number of clusters directly from KMeans
    initial_pi = np.zeros(k)
    initial_mean = np.zeros((k, d))
    initial_sigma = np.zeros((k, d, d))
    
    # ensure data is dataframe from np.ndarray
    data = pd.DataFrame(data=data)

    # get the number of iterations from the lenght of centroids (or labels)
    iterations = len(centroids)
        
    for i in range(iterations):
        iter_centroid = centroids[i]
        iter_labels = labels[i]
        # initialize counter to organize outputs per cluster k (counter = cluster_labels if the latter are int type ranging from 0 to k)
        counter=0
        for cluster_label in np.unique(iter_labels):
            # returns indices of rows estimated to belong to the cluster
            ids = np.where(iter_labels == cluster_label)[0] 
            # calculate pi (π = nk / n) for cluster k (πk)
            nk = data.iloc[ids].shape[0] # number of data points in current gaussian/cluster
            n = data.shape[0] # total number of points/rows in dataset
            initial_pi[counter] = nk / n

            # calculate mean (mu) of points estimated to be in cluster k (μk)
            initial_mean[counter,:] = np.mean(data.iloc[ids], axis = 0)
            de_meaned = data.iloc[ids] - initial_mean[counter,:]
            
            # calculate covariance (Σ, sigma) of points estimated to be in cluster k (Σk) 
            initial_sigma[counter, :, :] = np.dot(initial_pi[counter] * de_meaned.T, de_meaned) / nk
            
            counter+=1
        
        #assert np.sum(initial_pi) == 1    
            
    return (initial_pi, initial_mean, initial_sigma)

def initialise_parameters(data, k, iterations):
    """
    Calls the function KMeans to obtain the starting centroids and label values and use them as starting parameters for the EMGMM algorithm. 
    
    ------------    
    Parameters:
    
    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    
    ----------
    Returns:
    
    A tuple containing:
        
    - initial_pi: initial array of pik (πk = nk / n) values for each cluster k to input in the E-step of EM algorithm (shape: k,)
    - initial_mean: initial array of mu (μk, mean) values for each cluster k to input in the E-step of EM algorithm (shape: k, d)
    - initial_sigma: initial covariance matrices (Σk, sigma), one matrix for each cluster k to input in the E-step of EM algorithm (shape: k, d*d)
        
    """
    centroids, labels =  KMeans(data, k, iterations)

    (initial_pi, initial_mean, initial_sigma) = calculate_mean_covariance(data, centroids, labels)
        
    return (initial_pi, initial_mean, initial_sigma)

def e_step(data, pi, mu, sigma):
    """
    Performs E-step on GMM model
    
    ------------
    Parameters:
    
    - data: data points in numpy array (shape: n, d; where n: number of rows, d: number of features/columns)
    - pi: weights of mixture components pik (πk = nk / n) values for each cluster k in an array (shape: k,)
    - mu: mixture component mean (μk, mean) values for each cluster k in an array (shape: k, d)
    - sigma: mixture component covariance matrices (Σk, sigma), one matrix for each cluster k (shape: k, d, d)
    
    ----------
    Returns:

    - gamma: probabilities of clusters for datapoints (shape: n, k)

    """

    n = data.shape[0]
    k = len(pi)
    gamma = np.zeros((n, k))

    # convert np.nan to float(0.0)
    pi = np.nan_to_num(pi)
    mu = np.nan_to_num(mu)
    sigma = np.nan_to_num(sigma)

    x = data #.to_numpy() # Convert dataframe to np array

    for cluster in range(k):
        # Posterior Distribution using Bayes Rule
        gamma[ : , cluster] = pi[cluster] * multivariate_normal(mean=mu[cluster,:], cov=sigma[cluster], allow_singular=True).pdf(x)

    gamma = np.nan_to_num(gamma) # convert np.nan to float(0.0)
    # normalize across columns to make a valid probability
    gamma_norm = np.sum(gamma, axis=1)[ : , np.newaxis]
    gamma_norm = np.nan_to_num(gamma_norm)
    # avoid issues with divide by zero of ndarray
    gamma = np.divide(gamma, gamma_norm, out=np.zeros_like(gamma), where=gamma_norm!=0)
    #gamma /= gamma_norm

    return np.nan_to_num(gamma)

def m_step(data, gamma, sigma):
    """
    #TODO: further improve these docs

    Performs M-step of the GMM. It updates the priors, pi, means and covariance matrix.
    -----------
    Parameters:
    
    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    - gamma: probabilities of clusters for datapoints (shape: n, k)
    - sigma:
    
    ---------
    Returns:

    - pi: updated weights of mixture components pik (πk = nk / n) values for each cluster k in an array (shape: k,)
    - mu: updated mixture component mean (μk, mean) values for each cluster k in an array (shape: k, d)
    - sigma: updated mixture component covariance matrices (Σk, sigma), one matrix for each cluster k (shape: k, d, d)

    """
    n = data.shape[0] # number of datapoints (rows in the dataset), equivalent of gamma.shape[0]
    k = gamma.shape[1] # number of clusters
    d = data.shape[1] # number of features (columns in the dataset)
    
    # convert np.nan to float(0.0)
    gamma = np.nan_to_num(gamma)
    sigma = np.nan_to_num(sigma)

    # calculate pi and mu for each Gaussian
    pi = np.mean(gamma, axis = 0)
    # avoid issues with divide by zero of ndarray
    #mu = dividend / divisor
    dividend = np.dot(gamma.T, data)
    divisor = np.sum(gamma, axis = 0)[:,np.newaxis]
    mu = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)

    x = data #.to_numpy() # Convert dataframe to np array

    # update sigma for each Gaussian
    for cluster in range(k):
        x = x - mu[cluster, :] # (shape: n, d)
        gamma_diag = np.diag(gamma[: , cluster])
        x_mu = np.matrix(x)
        gamma_diag = np.matrix(gamma_diag)

        sigma_cluster = x.T * gamma_diag * x
        gamma = np.nan_to_num(gamma) # convert np.nan to float(0.0)
        #sigma = dividend / divisor
        dividend = (sigma_cluster)
        divisor = np.sum(gamma, axis = 0)[:,np.newaxis][cluster]
        sigma[cluster,:,:] = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)

    return pi, mu, sigma

def predict(data, pi, mu, sigma, k):
    """
    Returns predicted labels using Bayes Rule to calculate the posterior distribution
    
    -------------
    Parameters:
    
    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    - k: number of clusters to consider, 5 clusters by default

    ----------
    Returns:
    
    - labels: the predicted label/cluster based on highest probability gamma.
        
    """
    n = data.shape[0] # number of datapoints (rows in the dataset)
    labels = np.zeros((n, k))

    # convert np.nan to float(0.0)
    pi = np.nan_to_num(pi)
    mu = np.nan_to_num(mu)
    sigma = np.nan_to_num(sigma)
    
    #x = data.to_numpy() # Convert dataframe to np array

    for cluster in range(k):
        labels [:,cluster] = pi[cluster] * multivariate_normal(mean=mu[cluster,:], cov=sigma[cluster], allow_singular=True).pdf(data)
    labels  = labels .argmax(1)

    return labels 

 
def EMGMM(data, k, iterations, **kwargs):
    """
    Performs the Expectation-Maximisation (EM) algorithm to learn the parameters of a Gaussian mixture model (GMM), that is learning **π**, **μ** and **Σ**. A Gaussian mixture model (GMM) attempts to discover a mixture of multi-dimensional Gaussian probability distributions that best model any input dataset. GMMs can be used for finding clusters in the same manner as k-means.

    ------------
    Parameters:

    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    - k: number of clusters to consider, 5 clusters by default
    - n: number of iterations to consider, 10 iterations by default

    ------------
    Returns:

    The following csv files:

    - pi-[iteration].csv: This is a comma separated file containing the cluster probabilities of the EM-GMM model. The  k th row should contain the  k th probability,  πk , and there should be 5 rows. There should be 10 total files. For example, "pi-3.csv" will contain the cluster probabilities after the 3rd iteration.
    - mu-[iteration].csv: This is a comma separated file containing the means of each Gaussian of the EM-GMM model. The  k th row should contain the  k th mean , and there should be 5 rows. There should be 10 total files. For example, "mu-3.csv" will contain the means of each Gaussian after the 3rd iteration.
    - sigma-[cluster]-[iteration].csv: This is a comma separated file containing the covariance matrix of one Gaussian of the EM-GMM model. If the data is  d -dimensional, there should be  d  rows with  d  entries in each row. There should be 50 total files. For example, "sigma-2-3.csv" will contain the covariance matrix of the 2nd Gaussian after the 3rd iteration.

    """

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 1e-6
            
    d = data.shape[1] # number of features (columns in the dataset)
    n = data.shape[0] # number of datapoints (rows in the dataset)

    x = data #.to_numpy() # Convert dataframe to np array

    pi, mu, sigma =  initialise_parameters(data, k, iterations)

    for i in range(iterations):  
        gamma  = e_step(data, pi, mu, sigma)
        pi, mu, sigma = m_step(data, gamma, sigma)
        filename = "pi-" + str(i + 1) + ".csv"
        np.savetxt(filename, pi, delimiter=",") #this must be done at every iteration
        filename = "mu-" + str(i + 1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
        for cluster in range(k): #k is the number of clusters
            filename = "sigma-" + str(cluster + 1) + "-" + str(i + 1) + ".csv" #this must be done k times for each iteration
            np.savetxt(filename, sigma[cluster], delimiter=",")

    predicted_labels = predict(x, pi, mu, sigma, k)

    # convert np.nan to float(0.0)
    pi = np.nan_to_num(pi)
    mu = np.nan_to_num(mu)
    sigma = np.nan_to_num(sigma)

    # compute centers as point of highest density of distribution
    centroids = np.zeros((k,d))

    for cluster in range(k):
        density = multivariate_normal(mean=mu[cluster], cov=sigma[cluster], allow_singular=True).logpdf(x)
        centroids[cluster, :] = x[np.argmax(density)]
   
    return centroids, predicted_labels


def main():
    #Uncomment next line when running in Vocareum
    data=np.genfromtxt(sys.argv[1], delimiter = ',', skip_header=1)
    #Run KMeans to get clusters in the data and a csv of clusters per iteration
    centroids, labels = KMeans(data, 5, 10)
    #Run EMGMM to output the required csv files plus the predicted_labels
    c_emgmm, l_emgmm = EMGMM(data, 5, 10)

if __name__ == '__main__':
    main()