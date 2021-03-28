"""
This code implements a K-means and EM Gaussian mixture models per week 8 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.8
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

# import plotting modules; not needed in Vocareum
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TODO: General, check kmeans performance using sklearn library
# TODO: General, check GMM results using sklearn library

def KMeans(data, k:int = 5, n:int = 10, **kwargs):
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
    - n: number of iterations to consider, 10 iterations by default
    ------------
    Returns:

    - centroids, which are the means for each cluster {μ1,…,μK}
    - labels, are the corresponding assignments of each data point {c1,…,cn}, where each ci ∈ {1,…,K} and ci indicates which of the K clusters the observation xi belongs to.
    - writes the centroids of the clusters associated with each iteration to a csv file, one file per iteration; pass on a path if different from the default being "current working directory" + "outputs"
    """

    data = data.to_numpy() # Convert dataframe to np array
    centroids_list = []
    labels_list = []

    for i in range(n):
        centroids, label = kmeans2(data, k, iter = i+1, minit='points')
        centroids_list.append(centroids)
        labels_list.append(label)
        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        if 'path' in kwargs:
            path = kwargs['path']
            filepath = os.path.join(path, filename)
            np.savetxt(filepath, centroids, delimiter=",")
        else:
            path = os.path.join(os.getcwd(), "outputs")
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

def initialise_parameters(data):
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
    centroids, labels =  KMeans(data)

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

    x = data.to_numpy() # Convert dataframe to np array

    for cluster in range(k):
        # Posterior Distribution using Bayes Rule
        gamma[ : , cluster] = pi[cluster] * multivariate_normal(mean=mu[cluster,:], cov=sigma[cluster], allow_singular=True).pdf(x)

    gamma = np.nan_to_num(gamma) # convert np.nan to float(0.0)
    # normalize across columns to make a valid probability
    gamma_norm = np.sum(gamma, axis=1)[ : , np.newaxis]
    gamma_norm = np.nan_to_num(gamma_norm)
    gamma /= gamma_norm

    return np.nan_to_num(gamma)

def m_step(data, gamma, sigma):
    """
    Performs M-step of the GMM. It updates the priors, pi, means and covariance matrix.
    -----------
    Parameters:
    
    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    - gamma: probabilities of clusters for datapoints (shape: n, k)
    - sigma: mixture component covariance matrices (Σk, sigma), one matrix for each cluster k (shape: k, d, d)
    
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
    mu = np.dot(gamma.T, data) / np.sum(gamma, axis = 0)[:,np.newaxis]

    x = data.to_numpy() # Convert dataframe to np array

    # update sigma for each Gaussian
    for cluster in range(k):
        x = x - mu[cluster, :] # (shape: n, d)
        gamma_diag = np.diag(gamma[: , cluster])
        x_mu = np.matrix(x)
        gamma_diag = np.matrix(gamma_diag)

        sigma_cluster = x.T * gamma_diag * x
        gamma = np.nan_to_num(gamma) # convert np.nan to float(0.0)
        sigma[cluster,:,:] = (sigma_cluster) / np.sum(gamma, axis = 0)[:,np.newaxis][cluster]

    return pi, mu, sigma
       
def compute_loss_function(data, pi, mu, sigma, tol):
    """
    Computes the lower bound loss function for a given dataset, and values of pi, mu and sigma. 

    -----------  
    Parameters:
    
    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    - pi: weights of mixture components pik (πk = nk / n) values for each cluster k in an array (shape: k,)
    - mu: mixture component mean (μk, mean) values for each cluster k in an array (shape: k, d)
    - sigma: mixture component covariance matrices (Σk, sigma), one matrix for each cluster k (shape: k, d, d)
    - tol: tolerance for loss to be considered equivalent to zero and hence end iterative process (default 1e-6)

    -----------
    Returns:

    - loss_output: the value of lower bound loss function, as a float.
    
    """
    
    n = data.shape[0] # number of datapoints (rows in the dataset), equivalent of gamma.shape[0]
    k = len(pi) # number of clusters, equivalent to gamma.shape[1]
    loss = np.zeros((n, k)) # set up the array to store the loss values for each iteration

    x = data.to_numpy() # Convert dataframe to np array

    # convert np.nan to float(0.0)
    mu = np.nan_to_num(mu)
    pi = np.nan_to_num(pi)
    sigma = np.nan_to_num(sigma)

    for cluster in range(k):
        dist = multivariate_normal(mu[cluster], sigma[cluster], allow_singular=False)
        loss[:,cluster] = gamma[:,cluster] * (np.log(pi[cluster] + tol) + dist.logpdf(x) - np.log(gamma[:,cluster] + tol))
    
    loss = np.nan_to_num(loss) # convert np.nan to float(0.0)
    loss_output = np.sum(loss)

    return loss_output

def EMGMM(data, k:int = 5, iterations:int = 10, tol:float = 1e-6):
    """
    Performs the Expectation-Maximisation (EM) algorithm to learn the parameters of a Gaussian mixture model (GMM), that is learning **π**, **μ** and **Σ**. A Gaussian mixture model (GMM) attempts to discover a mixture of multi-dimensional Gaussian probability distributions that best model any input dataset. In the simple case, GMMs can be used for finding clusters in the same manner as k-means. For this model, we assume a generative process for the data as follows:

    xi|ci∼Normal(μci,Σci),ci∼Discrete(π)
    
    where:
    - the ith observation is first assigned to one of K clusters according to the probabilities in vector  π, and 
    - the value of observation xi is then generated from one of K multivariate Gaussian distributions, using the mean (μ) and covariance indexed by ci. 
    
    Finally, we implement the EM algorithm to maximize the equation below over all parameters (π,μ1,…,μK,Σ1,…,ΣK) using the cluster assignments (c1,…,cn) as the hidden data:
    
    p(x1,…,xn|π,μ,Σ)=∏ni=1p(xi|π,μ,Σ)

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
        
    d = data.shape[1] # number of features (columns in the dataset)
    n = data.shape[0] # number of datapoints (rows in the dataset)

    x = data.to_numpy() # Convert dataframe to np array

    # initialise outputs arrays as empty
    centroids_list = []
    predicted_labels_list = []
    
    pi, mu, sigma =  initialise_parameters(data)

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

        centroids_list.append(centroids)
        predicted_labels_list.append(predicted_labels)
   
    return centroids_list, predicted_labels_list


"""
Prediction functions to visualise performance of EMGMM algorithm
This is not needed in Vocareum
"""

def predict(data, pi, mu, sigma, k:int = 5):
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
    
def predict_proba(data, pi, mu, sigma, k:int = 5):
    """
    Returns the posterior probability of the predicted label/cluster for each datapoint.

    -------------
    Parameters:
    
    - data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd (shape: n, d; where n: number of rows, d: number of features/columns)
    - k: number of clusters to consider, 5 clusters by default
    
    ----------
    Returns:
    
    - post_proba: the posterior probability of the predicted label/cluster for each datapoint.
        
    """
    n = data.shape[0] # number of datapoints (rows in the dataset)
    post_proba = np.zeros((n, k))

    # convert np.nan to float(0.0)
    pi = np.nan_to_num(pi)
    mu = np.nan_to_num(mu)
    sigma = np.nan_to_num(sigma)
    
    #x = data.to_numpy() # Convert dataframe to np array
        
    for cluster in range(k):
        # Posterior Distribution using Bayes Rule, try and vectorise
        post_proba[:,cluster] = pi[cluster] * multivariate_normal(mean=mu[cluster,:], cov=sigma[cluster], allow_singular=True).pdf(data)
    
    return post_proba


"""
Helper functions:
"""
def write_csv(filename, outputdata, **kwargs):
    """
    Write the outputs to a csv file.

    ------------
    Parameters:

    - filename: name of the file to create
    - data: data to write to the output file
    - kwargs (optional): 
        - 'header': list of str for the headers to include in the outputs file created
        - 'path': str of the path to where the file needs to be created, specified if different from default ([./datasets/out])
    
    ------------
    Returns:

    - csv file with the output data

    """

    # 
    if 'header' in kwargs:
        header = kwargs['header']
    else:
        header = False
    if 'path' in kwargs:
        filepath = kwargs['path']
    else:
        filepath = os.path.join(os.getcwd(),'datasets','out', filename)

    df = pd.DataFrame(outputdata)
    df.to_csv(filepath, index = False, header = header)
    return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')

def get_data(filename, **kwargs):
    """
    Read data from a file given its name. Option to provide the path to the file if different from: [./datasets/in].
    ------------
    Parameters:
    - filename: name of the file to be read, to get the data from.
    - kwargs (optional): 
        - 'headers': list of str for the headers to include in the outputs file created
        - 'path': str of the path to where the file is read, specified if different from default ([./datasets/in])
    ------------
    Returns:
    - df: a dataframe of the data

    """
    
    # Define input filepath
    if 'path' in kwargs:
        filepath = kwargs['path']
    else:
        filepath = os.path.join(os.getcwd(),'datasets','out')

    input_path = os.path.join(filepath, filename)

    # If provided, use the title of the columns in the input dataset as headers of the dataframe
    if 'headers' in kwargs:
        # Read input data
        df = pd.read_csv(input_path, names = kwargs['headers'])
    else:
        # Read input data
        df = pd.read_csv(input_path)
       
    return df

def plot_simple(dataframe, filetitle:str = 'sampledata'):

    """
    Plots the sample data.
    ------------
    Parameters:

    ------------
    Returns:

    """

    df = dataframe

    names_axis = list(dataframe.columns)

    fig = go.Figure(data=go.Scatter(x=df[names_axis[0]], 
                                        y=df[names_axis[1]],
                                        mode='markers',
                                        marker=dict(
                                        line_width=1,
                                        size = 16),
                                        showlegend=False))  # turn off legend only for this item

    # Give the figure a title
    fig.update_layout(title='Kmeans | Clustering | Project 3',
                        xaxis_title=names_axis[0],
                        yaxis_title=names_axis[1])

    # Show the figure, by default will open a browser window
    fig.show()

def plot_clusters(dataframe, centroids, labels, **kwargs):
    """
    Plots the clusters identified in the data which vary per iteration. It also plots the centroids of the clusters considered by the kmeans algorithm for each iteration. The plot is presented on an interactive browser window which allows for each of the iterations to be turn on/off as desire by the user. 
    
    ------------
    Parameters:
    
    - dataframe: the dataframe of the points to plot (the original dataset)
    - centroids: x,y coordinates of the centroids of each cluster as obtained from KMeans algorithm
    - labels: labels of the cluster each centroid belongs to as obtained from KMeans algorithm
    - names_axis: names of the columns in the dataframe for naming the axis

    ------------
    Returns:
    - a seris of plots illustrating the sample data clustered (k number of clusters) and how the clustering varies per iteration. 
    """

    add_traces =[]

    df = dataframe

    names_axis = list(dataframe.columns)
   
    for i in range(len(centroids)):
        sample_data = go.Scatter(x=df[names_axis[0]], 
                                        y=df[names_axis[1]],
                                        name = 'iteration ' + str(i+1),
                                        mode='markers',
                                        marker=dict(
                                            color=labels[i],
                                            colorscale='Portland',
                                            line_width=1,
                                            size = 12),
                                        text=['cluster ' + str(k+1) for k in labels[i]], # hover text goes here
                                        legendgroup = 'iteration' + str(i+1),
                                        showlegend=True)  # turn off legend only for this item
        add_traces.append(sample_data)

        trace = go.Scatter(x=np.transpose(centroids[i])[0],
                            y=np.transpose(centroids[i])[1],
                            name='centroids iteration ' + str(i+1),
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                #color = k + 1,
                                #colorscale='Greys',
                                line_width=1,
                                size = 20,
                                ),
                            text=['centroid k=' + str(k + 1) for k in range(len(np.transpose(centroids[i])[0]))], # hover text goes here
                            legendgroup = 'centroid iteration' + str(i+1),
                            showlegend=True)  # turn off legend only for this item
        add_traces.append(trace)
        data = go.Data(add_traces)
        fig = go.Figure(data=data)

        # TODO: add circle/ellipses to represent extent of each KMeans / Gaussian
        #w_factor = 0.2 / model.pi.max()
        #for pos, covar, w in zip(model.mu, model.sigma, model.pi):
        #    draw_ellipse(pos, covar, alpha = w)

        # Give the figure a title
        if 'fig_type' in kwargs:
            fig_type = kwargs['fig_type']
            if fig_type == 'kmeans':
                fig_title='Kmeans | Clustering | Project 3 | k = ' + str(len(centroids[i])) + ' clusters, ' + 'total iterations i = ' + str(i + 1)
            if fig_type == 'gmm':
                fig_title='EM Gaussian Mixture Model (initialised from KMeans) | Clustering | Project 3 | k = ' + str(len(centroids[i])) + ' clusters, ' + 'total iterations i = ' + str(i + 1)
            fig.update_layout(title=fig_title,
                                xaxis_title=names_axis[0],
                                yaxis_title=names_axis[1])
        else:
            fig.update_layout(title='Clustering | Project 3 | k = ' + str(len(centroids[i])) + ' clusters, ' + 'total iterations i = ' + str(i + 1),
                                xaxis_title=names_axis[0],
                                yaxis_title=names_axis[1])

    # Show the figure, by default will open a browser window
    fig.show()

    return

"""
Main function
"""

def main():

    # Uncomment next line when running in Vocareum
    #data = np.genfromtxt(sys.argv[1], delimiter = ",")
    data = get_data('Clustering_gmm.csv', path = os.path.join(os.getcwd(), 'datasets'))
    #use for 2D clustering dataset sample
    plot_simple(data)
    k = 5
    n = iterations = 10
    # Run KMeans to get clusters in the data and a csv of clusters per iteration
    centroids, labels = KMeans(data, k, n)
    # Plot results after KMeans; for visualisation only, not required in assignment
    plot_clusters(data, centroids, labels, fig_type='kmeans')
    # Run EMGMM to output the required csv files plus the predicted_labels
    centroids_emgmm, labels_emgmm = EMGMM(data, k, 10)
    # Plot results after EMGMM; for visualisation only, not required in assignment
    plot_clusters(data, centroids_emgmm, labels_emgmm, fig_type='gmm')


if __name__ == '__main__':
    main()