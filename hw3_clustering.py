


import numpy as np
import pandas as pd
import scipy as sp
import sys

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


## To run the script, simply add this to the end of the starter code:

if __name__ == '__main__':

 data = np.genfromtxt(sys.argv[1], delimiter = ",")

 KMeans(data)

 EMGMM(data)

