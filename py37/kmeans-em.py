"""
This code implements a K-means and EM Gaussian mixture models per week 8 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.7
"""

# builtin modules
import sys
import os
import math
from random import randrange
import functools
import operator
import requests

# 3rd party modules
import numpy as np
import pandas as pd
import scipy as sp


def KMeans(X:list = X, k:int = 5, n_iter:int = 10):
	#perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively

    for i in range(n_iter):
	    filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        centerslist = 'test'
	    np.savetxt(filename, centerslist, delimiter=",")
    
    return

  
def EMGMM(X:list = X, k:int = 5, n_iter:int = 10):

	filename = "pi-" + str(i+1) + ".csv" 
	pi = 'test'
    np.savetxt(filename, pi, delimiter=",") 
	filename = "mu-" + str(i+1) + ".csv"
    mu = 'test'
	np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
    
  for j in range(k): #k is the number of clusters 
    filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
    np.savetxt(filename, sigma[j], delimiter=",")


def get_data(source_file, **kwargs):
    """
    Read data from a file given its name. Option to provide the path to the file if different from: [./datasets/in]

    """
    # Define input and output filepaths
    input_path = os.path.join(os.getcwd(),'datasets','in', source_file)

    if 'col_titles' in kwargs:
        # Read input data
        df = pd.read_csv(input_path, names = kwargs['col_titles'])
    else:
        # Read input data
        df = pd.read_csv(input_path)
       
    return df


def main():

	# for running in Vocareum
	#X = np.genfromtxt(sys.argv[1], delimiter = ",")
	X = np.genfromtxt(os.path.join(os.getcwd(),'datasets', '3D_spatial_network.txt'), delimiter=",")

	#data = np.genfromtxt(sys.argv[1], delimiter = ",")


	KMeans(X)
	
	EMGMM(X)

    ## write the results of the prediction to a csv
    np.savetxt("y_validate.csv", class_predicted, fmt='%1i', delimiter="\n") # write output to file, note values for fmt and delimiter



if __name__ == '__main__':
	main()