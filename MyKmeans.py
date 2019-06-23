#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy




from pyclustering.utils.metric import distance_metric, type_metric



class kmeans:
   
    
    
    def __init__(self, data, initial_centers,centers_index=None, tolerance = 0.001, ccore = True, **kwargs):
        """!
        @brief Constructor of clustering algorithm K-Means.
        @details Center initializer can be used for creating initial centers, for example, K-Means++ method.
        
        @param[in] data (array_like): Input data that is presented as array of points (objects), each point should be represented by array_like data structure.
        @param[in] initial_centers (array_like): Initial coordinates of centers of clusters that are represented by array_like data structure: [center1, center2, ...].
        @param[in] tolerance (double): Stop condition: if maximum value of change of centers of clusters is less than tolerance then algorithm stops processing.
        @param[in] ccore (bool): Defines should be CCORE library (C++ pyclustering library) used instead of Python code or not.
        @param[in] **kwargs: Arbitrary keyword arguments (available arguments: 'observer', 'metric').
        
        <b>Keyword Args:</b><br>
            - observer (kmeans_observer): Observer of the algorithm to collect information about clustering process on each iteration.
            - metric (distance_metric): Metric that is used for distance calculation between two points.
        
        @see center_initializer
        
        """
        self.__pointer_data = data
        self.__clusters = []
        self.__centers = numpy.matrix(initial_centers)
        self.__centers_index = centers_index
        self.__tolerance = tolerance

        
        self.__metric = kwargs.get('metric', distance_metric(type_metric.EUCLIDEAN_SQUARE))
        self.__metric.enable_numpy_usage()
        


    def process(self):
        """!
        @brief Performs cluster analysis in line with rules of K-Means algorithm.
        
        @remark Results of clustering can be obtained using corresponding get methods.
        
        @see get_clusters()
        
        """
        if self.__pointer_data[0].shape[0] != len(self.__centers[0]):
            raise NameError('Dimension of the input data and dimension of the initial cluster centers must be equal.')

        self.__clusters = self.__update_clusters()
   
    def get_clusters(self):
        """!
        @brief Returns list of allocated clusters, each cluster contains indexes of objects in list of data.
        
        @see process()
        
        """
        return self.__clusters

    def __update_clusters(self):
        """!
        @brief Calculate Euclidean distance to each point from the each cluster. Nearest points are captured by according clusters and as a result clusters are updated.
        
        @return (list) updated clusters as list of clusters. Each cluster contains indexes of objects from data.
        
        """
        clusters = [[] for _ in range(len(self.__centers))]
        
        if self.__centers_index is not None:
            
            if len(numpy.unique(self.__centers_index)) <  len(self.__centers_index) :
                print("\n Oppppss ")
                print(str(len(self.__centers_index))," >> ",self.__centers_index)
                raise NameError("Two centers is equal")
            cls = 0  
            for idx in self.__centers_index:
                clusters[cls].append(idx)
                cls+=1
        
        dataset_differences = numpy.zeros((len(clusters), self.__pointer_data.shape[0]))
        for index_center in range(len(self.__centers)):
            dataset_differences[index_center] = self.__metric(self.__pointer_data, self.__centers[index_center])
            
        optimum_indexes = numpy.argmin(dataset_differences, axis=0)
        
        for index_point in range(len(optimum_indexes)):
            if index_point not in (numpy.asarray(self.__centers_index)):
                index_cluster = optimum_indexes[index_point]
                clusters[index_cluster].append(index_point)
        
        
        
        return clusters
