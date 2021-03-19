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
from scipy.cluster.vq import kmeans2
from scipy.stats import multivariate_normal


# import plotting modules; not needed in Vocareum
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Plotting functions; not needed in Vocareum

def plot_inputs(df, names_in:list = ['A','B','label']):
        """
        Plot the input dataset as a scatter plot, showing the two classes with two different patterns.
        - source_file: csv file with the input samples
        - weights: from perceptron_classify function
        - names_in: a list of the names of the columns (headers) in the input df
        returns:
        - a plot of the figure in the default browser, and
        - a PNG version of the plot to the "images" project directory
        """ 
        # Create the figure for plotting the initial data
        fig = go.Figure(data=go.Scatter(x=df[names_in[0]], 
                                        y=df[names_in[1]],
                                        mode='markers',
                                        marker=dict(
                                        color=df[names_in[2]],
                                        colorscale='Viridis',
                                        line_width=1,
                                        size = 16),
                                        text=df[names_in[2]], # hover text goes here
                                        showlegend=False))  # turn off legend only for this item

        # Give the figure a title
        fig.update_layout(title='Perceptron Algorithm | Classification with support vector classifiers | Problem 3',
                          xaxis_title=names_in[0],
                          yaxis_title=names_in[1])

        # Show the figure, by default will open a browser window
        fig.show()

        # export plot to png file to images directory
        # create an images directory if not already present
        if not os.path.exists("images"):
            os.mkdir("images")
        ## write the png file with the plot/figure
        return fig.write_image("images/fig3.png")

def plot_model(X, y, xx, y_, Z, kernel_type:str):
        """
        Plot the decision boundary from:
        - X: the features dataset,
        - y: the labels vector, 
        - h: step size in the mesh, e.g. 0.02
        - grid_search: model of the grid_search already fitted
        - model_type: str of the type of model used for title of plot and filename of image to export
        returns:
        - a plot of the figure in the default browser, and
        - a PNG version of the plot to the "images" project directory
        """ 

        # Create the figure for plotting the model
        fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], 
                            mode='markers',
                            showlegend=False,
                            marker=dict(
                                color=y,
                                colorscale='Viridis',
                                line_width=1,
                                size = 16),
                            text='Label', # hover text goes here
                            showlegend=False))  # turn off legend only for this item
        
        # Add the heatmap to the plot
        fig.add_trace(go.Heatmap(x=xx[0], y=y_, z=Z,
                          colorscale='Rainbow',
                          showscale=False))
        
        # Give the figure a title and name the x,y axis as well
        fig.update_layout(
            title='Perceptron Algorithm | Classification with support vector classifiers | ' + kernel_type.upper(),
            xaxis_title='A',
            yaxis_title='B')

        # Show the figure, by default will open a browser window
        fig.show()

        # export plot to png file to images directory
        # create an images directory if not already present
        if not os.path.exists("images"):
            os.mkdir("images")
        ## write the png file with the plot/figure
        return fig.write_image("images/fig3-" + kernel_type + ".png")

def plot_multiple():
    """
    #TODO: Docs missing
    """
    # data
    np.random.seed(123)
    frame_rows = 50
    n_plots = 36
    frame_columns = ['V_'+str(e) for e in list(range(n_plots+1))]
    df = pd.DataFrame(np.random.uniform(-10,10,size=(frame_rows, len(frame_columns))),
                      index=pd.date_range('1/1/2020', periods=frame_rows),
                        columns=frame_columns)
    df=df.cumsum()+100
    df.iloc[0]=100

    # plotly setup
    plot_rows=6
    plot_cols=6
    fig = make_subplots(rows=plot_rows, cols=plot_cols)

    # add traces
    x = 0
    for i in range(1, plot_rows + 1):
        for j in range(1, plot_cols + 1):
            #print(str(i)+ ', ' + str(j))
            fig.add_trace(go.Scatter(x=df.index, y=df[df.columns[x]].values,
                                     name = df.columns[x],
                                     mode = 'lines'),
                         row=i,
                         col=j)

            x=x+1

    # Format and show fig
    fig.update_layout(height=1200, width=1200)
    fig.show()

    return

# Not needed
def summarize_dataframe(dataframe, class_value, n_features):

    """
    Calculate the mean, standard deviation and count for each column in the dataframe from the following inputs:
        dataframe : dataset to summarise as a DataFrame
        class_value : the value (label from 0 to 9) of the class being summarised
        n_features : number of features (columns) in the training dataset (X_train + y_train)
    
    It returns a DataFrame of mean, std and count for each column/feature in the dataset. The number of features is used to populate the mean, stdv and coun figures for the unseen classes in the training dataset for the number of classes specify in k_classes.

    """
    if dataframe.shape == (0,0):
        mean = np.append(np.zeros(n_features), [class_value])
        sigma = np.zeros(n_features + 1)
        count = np.zeros(n_features + 1)
    else:
        mean = dataframe.mean(axis=0)
        sigma = dataframe.std(axis=0, ddof=1)  #ddof = 0 to have same behaviour as numpy.std, std takes the absolute value before squaring
        count = dataframe.count(axis=0)
    
    frame = {'mean': mean, 'std': sigma, 'count': count}

    summaries = pd.DataFrame(frame)

    return summaries


def KMeans(data, k:int = 5, n:int = 10, **kwargs):
	"""
	It uses the scipy.klearn2 algorithm to classify a set of observations into k clusters using the k-means algorithm. It attempts to minimize the Euclidean distance between observations and centroids. Note the following methods for initialization are available:
	# - ‘random’: generate k centroids from a Gaussian with mean and variance estimated from the data.
	# - ‘points’: choose k observations (rows) at random from data for the initial centroids.
	# - ‘++’: choose k observations accordingly to the kmeans++ method (careful seeding)
	# - ‘matrix’: interpret the k parameter as a k by M (or length k array for 1-D data) array of initial centroids.

	It is recommended the kmeans algorithm is initialized by randomly selecting 5 data points (minit = "points", with k = 5)

    Also, note we use the identity matrix for each cluster's covariance matrix for the initialization. We also try initializing data points without replacement: for example, using np.random.choice and set replace = False.
	
	Parameters:
	- data: ndarray of the set of observations to classify, the n data points {x1,…,xn}, where each xi ∈ Rd
	- k: number of clusters to consider, 5 clusters by default
	- n: number of iterations to consider, 10 iterations by default
	
	Returns:
    - centroids, which are the means for each cluster {μ1,…,μK}
    - labels, are the corresponding assignments of each data point {c1,…,cn}, where each ci ∈ {1,…,K} and ci indicates which of the K clusters the observation xi belongs to.
	- writes the centroids of the clusters to a txt file; pass on a path if different from the default being "current working directory" + "outputs" 
	
	"""
	# Convert dataframe to np array
	data = data.to_numpy()

    # TODO: Before passing array of data to kmeans2, we need to convert it to a Multivariate Gaussian distribution so 'data' becomes an array with shape (100, 2), where the first column is the mean, and second one the covariance; each row is each 'ith' observation in the dataset. Note the covariance shall be the identity matrix to start with. The means are the centroids.
    # mean = centroids
    # cov = np.identity(n); n = shape of matrix to match columns in mean/centroids vector
    #Multivariate_Gaussian = np.random.multivariate_normal(mean, cov)

	# Use scipy method 'kmeans2' for obtaining the centroids of clusters in data
	# We use the 'kmeans2' method inside a loop to save results from each iteration
    # TODO: include comparison to scipy.cluster.vq.kmeans
    
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


def EMGMM(data):
	"""
	Performs the Expectation-Maximisation (EM) algorithm to learn the parameters of a Gaussian mixture model (GMM), that is learning **π**, **μ** and **Σ**. For this model, we assume a generative process for the data as follows:

	xi|ci∼Normal(μci,Σci),ci∼Discrete(π)
	
	where:
   - the ith observation is first assigned to one of K clusters according to the probabilities in vector  π, and 
   - the value of observation xi is then generated from one of K multivariate Gaussian distributions, using the mean (μ) and covariance indexed by ci. 
	
	Finally, we implement the EM algorithm to maximize the equation below over all parameters (π,μ1,…,μK,Σ1,…,ΣK) using the cluster assignments (c1,…,cn) as the hidden data:
	
	p(x1,…,xn|π,μ,Σ)=∏ni=1p(xi|π,μ,Σ)
	"""
	filename = "pi-" + str(i+1) + ".csv" 
	np.savetxt(filename, pi, delimiter=",") 
	filename = "mu-" + str(i+1) + ".csv"
	np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
    
  for j in range(k): #k is the number of clusters 
    filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
    np.savetxt(filename, sigma[j], delimiter=",")


def write_csv(filename, a, **kwargs):
        # write the outputs csv file
        if 'header' in kwargs:
            header = kwargs['header']
        else:
            header = False
        if 'path' in kwargs:
            filepath = kwargs['path']
        else:
            filepath = os.path.join(os.getcwd(),'datasets','out', filename)

        df = pd.DataFrame(a)
        df.to_csv(filepath, index = False, header = header)
        return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')


def get_data(source_file, **kwargs):
    """
    Read data from a file given its name. Option to provide the path to the file if different from: [./datasets/in]

    """
    # Define input and output filepaths
    input_path = os.path.join(os.getcwd(),'datasets', source_file)

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
	#data = np.genfromtxt(sys.argv[1], delimiter = ",")
	#X = pd.DataFrame(data=X)
	#data = pd.DataFrame(data=data)

	df = get_data('iris.data.csv', col_titles=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species_class']) #use for irisdata set

	# to ensure label is numerical, convert the last column of the dataframe to numerical instead of categorical; note this will not be needed in Vocareum as data passed onto functions is expected to be non-categorical already.	
	_, cols = df.shape
	df[df.columns[cols - 1]] = df[df.columns[cols - 1]].astype('category')
	col_class = df.select_dtypes(['category']).columns
	df[col_class] = df[col_class].apply(lambda x: x.cat.codes)

	data = df
	
	# write the clusters in the data, one txt per iteration
	KMeans(data)
	
	EMGMM(data)


if __name__ == '__main__':
	main()