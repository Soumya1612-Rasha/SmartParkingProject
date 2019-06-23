#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from functools import partial
from scipy import sparse # import sparse module from SciPy package
import logging
from time import perf_counter
from sklearn import preprocessing
from colors import red, green, blue, cyan
from collections import Counter
from Configuration import Cfg
import sys


# -----------------------------------------------------------------------------
# Park clustering metric
# -----------------------------------------------------------------------------
def clust_park_metric( u, v, W,k, m=0,Y=None, Y_norm_squared=None,
                         squared=False):
    """
    Custom metric function for clustering problem
    """

    if m==0:
        return numpy.sqrt(numpy.sum(numpy.square(u-v)))
    elif m==1:
        #print(numpy.abs((u-v)).dot(  k * numpy.abs(W).max(axis=1) ) )
        return ( numpy.abs((u-v)).dot(  k * numpy.abs(W).max(axis=1) ) )
    elif m==2:
        return ( numpy.abs((u-v)).dot(  ( numpy.abs(W).max(axis=1) + k) ) )
    elif m==3:
        return ( numpy.abs((u-v)).dot(  k * numpy.abs(W).mean(axis=1) ))
    elif m==4:
        return ( numpy.abs((u-v)).dot(  ( numpy.abs(W).mean(axis=1) + k) ) )
    else :
        raise ValueError("m must be in {0,1,2,3,4}")

# -----------------------------------------------------------------------------
# Clustering_tools
# -----------------------------------------------------------------------------

class Clustering_tools(object):
    """
    Clustering tools for parking 
    """
    def __init__(self, data_clust, clusters_columns):
        """
        Constructor
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_clust = data_clust
        self.clusters_columns = clusters_columns
        self.logger.debug(blue("Clusters are : ")
                          + str(self.clusters_columns))
        self.nb_clust = len(clusters_columns)
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(list(range(0,self.nb_clust)))
            

    # _________________________________________________________________________
    def kmeans_v2(self,data_clust,data_test,X,X_test
                        , init='k-means++'):
        """
        Clustering of data using KMeans_v2 approach
        input data_clust : dataframe
        input data_test : dataframe
        input centers : array of centers
        return Xlabled,centers,labels,X_testlabled,labels_test,time

        """
        
        t0 = perf_counter()  # timer
        from sklearn.cluster import KMeans
        kmeans_v2 = KMeans(n_clusters=self.nb_clust, init=init
                         , random_state=Cfg.RandomState
                           , n_jobs=-1)
        labels = kmeans_v2.fit_predict(data_clust)
        centers = kmeans_v2.cluster_centers_  # centers
        
        labels = labels.reshape(data_clust.shape[0],1)
        self.logger.info(cyan(" Kmeans-Train-Clustering: "+str(Counter(list(labels[:,0])))))
        labels_copy = self.lb.transform(labels)
        if Cfg.Normalize:
            labels_copy =  preprocessing.normalize(labels_copy,
                                axis=0, norm='l2')
        X = sparse.hstack([X[:,0:Cfg.NbZones],data_clust
                           ,labels_copy],format="csr")
    
        # Labels prediction for test set 
        labels_test = None
        if X_test.shape[0] > 0:
            labels_test = kmeans_v2.predict(data_test) 
            labels_test = labels_test.reshape(data_test.shape[0],1)
            self.logger.info(cyan(" Kmeans-Test-Clustering: "+str(Counter(list(labels_test[:,0])))))
            labels_test_copy = self.lb.transform(labels_test)
            if Cfg.Normalize:
                labels_test_copy = preprocessing.normalize(labels_test_copy,
                                        axis=0, norm='l2')
            X_test = sparse.hstack([X_test[:,0:Cfg.NbZones],data_test,
                                    labels_test_copy],format="csr")    
              
        return X,centers,labels,X_test,labels_test,(round(perf_counter()-t0,2))
    
  
    # _________________________________________________________________________
    def CALM(self, data_clust, data_test
                     , X, X_test,y_train,
                        model,labels_train, W=None, k=0, m=0):
        """
        Clustering of data using CALM approach
        input W : array of weights
        input data_clust : sparse matrix
        input data_test : sparse matrix
        input X_train : sparse matrix
        input X_test : sparse matrix
        input model : DNN to predict
        input k : float
        input m : {0.1.2} costum metric 
        input init : array of centers
        return Xlabled,centers,labels,X_testlabled,labels_test,time

        """
        # calcul of new centers
        #<<
        t0 = perf_counter()  # timer
        db =[]
        
        for c in range(self.nb_clust):
            db.append([])
        for i in range(len(labels_train)):
            db[int(labels_train[i])].append(i)
       
        db = [eldb for eldb in db if len(eldb) > 0]
        
        
        centers_index = []
        
        for c in range(0,len(db)):
            Y_t = model.predict(X[db[c],:])
            errors = numpy.abs(Y_t-y_train[db[c],0].toarray())
            index_min = numpy.argmin(errors)
            centers_index.append(db[c][index_min])
        
        
        centers = data_clust[centers_index,:]
        self.logger.debug(green(' Calcul of centers took: ') 
                         + red(str(round(perf_counter()-t0,5))) 
                         + " " +green(".s") )
        #>>
        
        
        from pyclustering.utils.metric import type_metric, distance_metric
        from MyKmeans import kmeans
        
        
        custom_metric = partial(clust_park_metric, W=W,k=k,m=m)
        
        metric = distance_metric(type_metric.USER_DEFINED
                                 , func=custom_metric)
        
        kmeans_v3 = kmeans(data_clust, centers.toarray()
                                ,centers_index = centers_index
                               , metric=metric)
        kmeans_v3.process()
        clusters = kmeans_v3.get_clusters()
        labels = numpy.zeros((data_clust.shape[0],1 ),dtype=int)
        
        # get labels from clusters
        
            
        cls = 0
        for cluster_idx in range(len(clusters)):
            labels[clusters[cluster_idx]]= int(cls)
            cls+=1
        #self.logger.info(cyan(" CALM-Train-Clustering: "+str(Counter(list(labels[:,0])))))
        labels_copy = self.lb.transform(labels)
        if Cfg.Normalize:
            labels_copy =  preprocessing.normalize(labels_copy,
                                axis=0, norm='l2')
        X = sparse.hstack([X[:,0:Cfg.NbZones],data_clust
                           ,labels_copy],format="csr")
        
        # Labels prediction for test set 
        labels_test = None
        if X_test.shape[0] > 0:
            
            kmeans_v3 = kmeans(data_test
                                , centers.toarray()
                                , metric=metric)
            kmeans_v3.process()
            clusters = kmeans_v3.get_clusters()
            
            labels_test = numpy.zeros((data_test.shape[0],1),dtype=int)
            # get labels from clusters
            cls = 0
            for cluster_idx in range(len(clusters)):
                labels_test[clusters[cluster_idx]]= int(cls)
                cls+=1
            
            #self.logger.info(cyan(" CALM-Test-Clustering: "+str(Counter(list(labels_test[:,0])))))
            labels_test_copy = self.lb.transform(labels_test)
            if Cfg.Normalize:
                labels_test_copy = preprocessing.normalize(labels_test_copy,
                                        axis=0, norm='l2')
            X_test = sparse.hstack([X_test[:,0:Cfg.NbZones],data_test,
                                    labels_test_copy],format="csr")  
        
        
        return X,labels,X_test,labels_test,(round(perf_counter()-t0,2))
    
     # _________________________________________________________________________
    def Random(self, data_clust, data_test, X_train, X_test):
        """
        Clustering of data using Random approach
        input W : array of weights
        input data_clust : sparse matrix
        input data_test : sparse matrix
        input X_train : sparse matrix
        input X_test : sparse matrix
        return Xlabled,centers,labels,X_testlabled,labels_test,time

        """
        t0 = perf_counter()
        # Labels for train set
        
        labels = numpy.random.choice(range(0,self.nb_clust)
                            , data_clust.shape[0]
                            , replace=True).reshape(data_clust.shape[0],1)
        labels_copy = self.lb.transform(labels)
        if Cfg.Normalize:
            labels_copy =  preprocessing.normalize(labels_copy,
                                axis=0, norm='l2')
        X = sparse.hstack([X_train[:,0:Cfg.NbZones],data_clust
                           ,labels_copy],format="csr")
        
        
        # Labels prediction for test set 
        
        labels_test = None
        if X_test.shape[0] > 0:
            labels_test = numpy.random.choice(range(0,self.nb_clust)
                                , data_test.shape[0]
                                , replace=True).reshape(data_test.shape[0],1) 
            
            labels_test_copy = self.lb.transform(labels_test)
            if Cfg.Normalize:
                labels_test_copy = preprocessing.normalize(labels_test_copy,
                                        axis=0, norm='l2')
            X_test = sparse.hstack([X_test[:,0:Cfg.NbZones],data_test,
                                    labels_test_copy],format="csr")             
    
        return X,labels,X_test,labels_test,(round(perf_counter()-t0,2))
