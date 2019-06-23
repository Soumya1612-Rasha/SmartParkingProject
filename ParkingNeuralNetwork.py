#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from Configuration import Cfg
from colors import red, green, cyan
from time import perf_counter




# -----------------------------------------------------------------------------
# Parking_neural_network
# -----------------------------------------------------------------------------
class Parking_neural_network(object):
    """
    Parking Neural Network
    """
    def __init__(self):
        """
        Constructor
        """
        self.trained = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        
    # _________________________________________________________________________
    def prepare_data_for_clust(self, X):
        """
        Remove clustering variable from data
        """
        columns_clust = []
        for cl in range(1,Cfg.NbClusters+1):
            columns_clust.append('CLUSTER_'+ str(cl))
        X_clust = X[:,Cfg.NbZones:X.shape[1]-Cfg.NbClusters]
        return X_clust, columns_clust
    
    # _________________________________________________________________________
    def create_static_model(self, X_train):
        """
        Create the static Neural Network Model
        """
        from keras import models
        from keras import layers
        
        model = models.Sequential()    
        model.add(layers.Dense(64
                            , input_shape=(X_train.shape[1],)
                            , activation= "relu"
                            , kernel_initializer="uniform"))
        model.add(layers.Dense(32
                            , activation= "relu"
                            , kernel_initializer="uniform"))
        model.add(layers.Dense(8
                            , activation= "relu"
                            , kernel_initializer="uniform"))
        model.add(layers.Dense(4
                            , activation= "relu"
                            , kernel_initializer="uniform"))
        model.add(layers.Dense(1
                            , activation= "relu"
                            , kernel_initializer="uniform"))
        model.compile(optimizer='adam',
                      loss='mean_absolute_error')
        
        self.model = model

    # _________________________________________________________________________
    def search_best_model(self,data):
        
        """
        Search the best model using hyperopt & stock best model 
        input data: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from hyperopt import Trials, tpe,rand
        from hyperas import optim
        
        t0 = perf_counter() # timer 
        best_run, best_model,space = optim.minimize(model= create_model, 
                                          data=data,
                                          rseed=Cfg.RandomState,
                                          algo=tpe.suggest,
                                          max_evals=Cfg.NbTrials,
                                          trials=Trials(),
                                          eval_space=True,
                                          verbose=False,
                                          return_space=True
                                          )
        self.model = best_model
        self.best_run = best_run
        self.logger.info(' best-run:  '+cyan(str(best_run)))
        self.logger.info(green(' The serach of best model took: ') 
                         + red(str(round(perf_counter()-t0,5))) 
                         + " " +green(".s") )
    
    # _________________________________________________________________________
    def train(self,X,Y):
        """
        Train the Neural Network
        """
        t0 = perf_counter() # timer 
        batch_size = Cfg.batch_size
        if Cfg.dynamic==1:
            batch_size = self.best_run["batch_size"] 
        history = self.model.fit(X,Y
                                 ,epochs=Cfg.epochs
                                 ,batch_size=batch_size
                                 ,verbose=1,shuffle=False) 
        self.trained = 1
        return history,round(perf_counter()-t0,5)
        
    # _________________________________________________________________________
    def getW(self):
        """
        Get weights of layer 1 after training of the model
        """
        if self.trained == 0:
            raise Exception("Can not get weights of no trained PNN")
        W = self.model.layers[0].get_weights()[0]
        #print()
        #for layer in self.model.layers:
        #    print(layer.get_output_at(0).get_shape().as_list())
        #sys.stdin.read(1)
        # biases = self.model.layers[0].get_weights()[1]
        return W

    # _________________________________________________________________________
    def predict(self, X):
        """
        Prediction of new X
        """
        Y_test_hat = self.model.predict(X.as_matrix())
        return Y_test_hat
    

# _____________________________________________________________________________
def create_model( X_train, y_train, X_val, y_val):
    """
    Create the Neural Network Model
    """
    from hyperopt import  STATUS_OK
    from hyperas.distributions import choice
    from keras import models
    from keras import layers
    
    model = models.Sequential()    
    #==================================================================
    
    # Layer in
    model.add(layers.Dense(64
                        , input_shape=(X_train.shape[1],)
                        , activation= "relu"
                        , kernel_initializer="uniform"))
       
    #==================================================================
    # Layer 
    model.add(layers.Dense(32
                        , activation= "relu"
                        , kernel_initializer="uniform"))
    
    
    #==================================================================
    # Layer 
    model.add(layers.Dense(8
                        , activation= "relu"
                        , kernel_initializer="uniform"))
    
    
    #==================================================================
    # Layer 
    model.add(layers.Dense(4
                        , activation= "relu"
                        , kernel_initializer="uniform"))
    
   
    #==================================================================
    # Layer out
    model.add(layers.Dense(1
                        , activation= "relu"
                        , kernel_initializer="uniform"))
    #==================================================================
    
    model.compile(optimizer='adam',
                  loss='mean_absolute_error')
    #==================================================================
    # fit
    model.fit(X_train,
              y_train,
              epochs=1,
              batch_size={{choice([32, 32])}},verbose=0,
              shuffle=False
              )
    # evaluate
    score = model.evaluate(X_val, y_val, verbose=0)
    
    return {'loss': score, 'status': STATUS_OK, 'model': model}
