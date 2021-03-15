"""
This code implements a K-means and EM Gaussian mixture models per week 8 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.X for running on Vocareum

Execute as follows:
$ python3 hw3_classification.py X.csv
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

X = np.genfromtxt(sys.argv[1], delimiter = ",")


def KMeans(data):
	#perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively

	filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
	np.savetxt(filename, centerslist, delimiter=",")

  
def EMGMM(data):

	filename = "pi-" + str(i+1) + ".csv" 
	np.savetxt(filename, pi, delimiter=",") 
	filename = "mu-" + str(i+1) + ".csv"
	np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
    
  for j in range(k): #k is the number of clusters 
    filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
    np.savetxt(filename, sigma[j], delimiter=",")


def main():

	# for running in Vocareum
	X = np.genfromtxt(sys.argv[1], delimiter = ",")
	data = np.genfromtxt(sys.argv[1], delimiter = ",")

	KMeans(data)
	
	EMGMM(data)


if __name__ == '__main__':
	main()