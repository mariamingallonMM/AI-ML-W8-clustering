"""
This code implements a K-means and EM Gaussian mixture models per week 8 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.7 for running on Vocareum
Version run on Vocareum for grading (all docs removed to avoid issues with grading platform)

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
            # calculate pi = nk / n for cluster k
            nk = data.iloc[ids].shape[0] # number of data points in current gaussian/cluster
            n = data.shape[0] # total number of points/rows in dataset
            initial_pi[counter] = nk / n

            # calculate mean (mu) of points estimated to be in cluster k (k)
            initial_mean[counter,:] = np.mean(data.iloc[ids], axis = 0)
            de_meaned = data.iloc[ids] - initial_mean[counter,:]
            
            # calculate covariance (sigma) of points estimated to be in cluster k (k) 
            initial_sigma[counter, :, :] = np.dot(initial_pi[counter] * de_meaned.T, de_meaned) / nk
            
            counter+=1
        
        #assert np.sum(initial_pi) == 1    
            
    return (initial_pi, initial_mean, initial_sigma)

def initialise_parameters(data, k, iterations):
    
    centroids, labels =  KMeans(data, k, iterations)

    (initial_pi, initial_mean, initial_sigma) = calculate_mean_covariance(data, centroids, labels)
        
    return (initial_pi, initial_mean, initial_sigma)

def e_step(data, pi, mu, sigma):

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
            filename = "Sigma-" + str(cluster + 1) + "-" + str(i + 1) + ".csv" #this must be done k times for each iteration
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