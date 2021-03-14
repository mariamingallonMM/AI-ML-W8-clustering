# AI-ML-W8-clustering

ColumbiaX CSMM.102x Machine Learning Course. Week 8 Assignment on clustering. In this project we implement the K-means and EM Gaussian mixture models.


## Instructions

In this assignment, we will implement the K-means and EM Gaussian mixture models. Assume the following:

- We are given **n** data points **{x1,…,xn}**, where each **xi∈Rd**.
- With **K-means** we are trying to find **K centroids {μ1,…,μK}** and the corresponding assignments of **each data point {c1,…,cn}**, where each **ci ∈ {1,…,K}** and ci indicates which of the K clusters the observation xi belongs to. The objective function that we seek to minimize can be written:
![equation_1: L=∑ni=1∑Kk=11(ci=k)∥xi−μk∥2.](./ref/eq1.JPG?raw=true)
- We will use the **EM algorithm** (Expectation-Maximisation algorithm) to learn the parameters of a **Gaussian mixture model** (GMM), that is learning **π**, **μ** and **Σ**. For this model, we assume a generative process for the data as follows:
![equation_2: xi|ci∼Normal(μci,Σci),ci∼Discrete(π).](./ref/eq2.JPG?raw=true)
- In the above equation, the  *ith* observation is first assigned to one of  **K  clusters** according to the probabilities in **vector  π**, and the value of observation  **xi**  is then generated from one of  **K multivariate Gaussian distributions**, using the *mean (μ)* and *covariance indexed by ci*. 
- Finally, implement the EM algorithm to maximize, the equation below over all parameters **π,μ1,…,μK,Σ1,…,ΣK** using the cluster assignments **c1,…,cn** as the hidden data:
![equation_3: p(x1,…,xn|π,μ,Σ)=∏ni=1p(xi|π,μ,Σ).](./ref/eq3.JPG?raw=true)
- Note that the **K-means** and **EG-GM** algorithms shall be written to **learn 5 clusters**. Run both algorithms for **10 iterations**. The algorithms can be initialized arbitrarily. It is recommended that the K-means centroids are initialized by randomly selecting 5 data points ([K-memoids](https://en.wikipedia.org/wiki/K-medoids)). For the EM-GMM, the mean vectors can be initialized in the same fashion, with **π** initialized to be the uniform distribution and each **Σk** to be the identity (I) matrix.

More details about the inputs and the expected outputs are given below.

## Execute the program

The following command will execute your program:
`$ python3 hw3_clustering.py X.csv`

Note the following:
- The name of the dataset is passed on via `X.csv`. This file is a comma separated file containing the data. Each row corresponds to a single vector xi .
- The main .py file shall be named `hw3_clustering.py`. This file includes both the K-means and the EM-GMM algorithms.

You should write your K-means and EM-GMM codes to learn 5 clusters. Run both algorithms for 10 iterations. You can initialize your algorithms arbitrarily. We recommend that you initialize the K-means centroids by randomly selecting 5 data points. For the EM-GMM, we also recommend you initialize the mean vectors in the same way, and initialize  π  to be the uniform distribution and each  Σk  to be the identity matrix. 


## Expected Outputs from the program

When executed, the code writes the output to the file listed below following the formatting requirements specified also below.

When executed, the code writes several output files, each as described below. Note the formatting instructions given below. Where [iteration] and [cluster] are noted below, these shall be replaced with the iteration number and the cluster number.

- centroids-[iteration].csv: This is a comma separated file containing the K-means centroids for a particular iteration. The  k th row should contain the  k th centroid, and there should be 5 rows. There should be 10 total files. For example, "centroids-3.csv" will contain the centroids after the 3rd iteration.
- pi-[iteration].csv: This is a comma separated file containing the cluster probabilities of the EM-GMM model. The  k th row should contain the  k th probability,  πk , and there should be 5 rows. There should be 10 total files. For example, "pi-3.csv" will contain the cluster probabilities after the 3rd iteration.
- mu-[iteration].csv: This is a comma separated file containing the means of each Gaussian of the EM-GMM model. The  k th row should contain the  k th mean , and there should be 5 rows. There should be 10 total files. For example, "mu-3.csv" will contain the means of each Gaussian after the 3rd iteration.
- Sigma-[cluster]-[iteration].csv: This is a comma separated file containing the covariance matrix of one Gaussian of the EM-GMM model. If the data is  d -dimensional, there should be  d  rows with  d  entries in each row. There should be 50 total files. For example, "Sigma-2-3.csv" will contain the covariance matrix of the 2nd Gaussian after the 3rd iteration.


## Note on Correctness

Please note that for both of these problems, there are multiple potential answers depending on your initialization. However, the K-means and EM-GMM algorithms have known deterministic properties that we discussed in class, and so in this sense we can distinguish between correct and incorrect answers. We strongly suggest that you test out your code on your own computer before submitting. The UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/) has a good selection of datasets for clustering.


## Notes on data repositories

The following datasets have been selected from the UCI Machine Learning Repository for use and testing of the code written for this assignment:

- [Forest Fires Data Set](http://archive.ics.uci.edu/ml/datasets/Forest+Fires). This is a difficult regression task, where the aim is to predict the burned area of forest fires, in the northeast region of Portugal, by using meteorological and other data (see details [here](http://www.dsi.uminho.pt/~pcortez/forestfires)).
- [Wine Quality Data Set](http://archive.ics.uci.edu/ml/datasets/Wine+Quality). Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009](http://www3.dsi.uminho.pt/pcortez/wine/)).
- [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris). This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (see [Duda & Hart](http://rexa.info/paper/e6b7a3a8c46efef785a6ab735be07dafa0713ff3), for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

## Citations & References

- [Forest Fires Data Set](http://archive.ics.uci.edu/ml/datasets/Forest+Fires) by P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9.
- [Wine Quality Data Set](http://archive.ics.uci.edu/ml/datasets/Wine+Quality) by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 dataset has been used for testing the Naives Bayes classifier as it consists of colour images in 10 different classes. Chapter 3 of the following report describes the dataset and the methodology followed when collecting the CIFAR-10 dataset in much greater detail. Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
- [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris) by Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
