# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import logging
import sys
from colors import   cyan, red, green, blue, yellow
import matplotlib.pyplot as plt
from ParkingNeuralNetwork import Parking_neural_network
from ClusteringTools import Clustering_tools
import numpy

from scipy import sparse # import sparse module from SciPy package
#numpy.set_printoptions(threshold=numpy.nan)
import argparse
from Configuration import Cfg
from pandas.tools.plotting import parallel_coordinates
import pandas 

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#   Welcome message
#  ----------------------------------------------------------------------------
def print_welcome(args):
    print(red("_____________________________________________________________"))
    print(red("_____________________________________________________________\n"))
    print(cyan("     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"))
    print(cyan("         ^^^^^^^ ")
                    +yellow("Parking Neural Network")
                    +cyan("  ^^^^^"))
    print(cyan("             ^^^^      ")+yellow("-----")+cyan("           ^^^^"))
    print(cyan("                 ^^^^^^^^^^^^^^^^^^^^^^^   \n"))
    print(blue(args))
    print(red("_____________________________________________________________"))
    print(red("_____________________________________________________________\n"))
# -----------------------------------------------------------------------------
#   Menu
#  ----------------------------------------------------------------------------
def Menu():
    """
        Create args type and value
        return argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, required=True
                        , help='Number of iterations')
    parser.add_argument('-instn', type=int,default=1
                        , required=False
                        , help='Instance number')
    parser.add_argument('-gen', type=int,default=0
                        , required=False
                        , help='Generate and exit if gen == 1')
    parser.add_argument('-r', type=int,default=10000000000000000
                        , required=False, help='Number of data rows' )
    parser.add_argument('-c', type=int,default=0, required=False
                        , help='Number of clusters' )
    parser.add_argument('-ph', type=int,default=0, required=False
                        , help='Plot or not training history {0,1}' )
    parser.add_argument('-nz', type=int, required=True
                        , help='Number of zones' )
    parser.add_argument('-rs', type=int,default=0
                        , required=False
                        , help='Random_state value' )
    parser.add_argument('-nr', type=int, default=0, required=False
                        , help='Normalize data or not {0,1}' )
    
    parser.add_argument('-cm', type=int,default=0
                        , required=False
                        , help='Clustering method {0: standard, 1 : CALM, 2: Random}' )
    parser.add_argument('-nt', type=int,default=4
                        , required=False
                        , help='Number of trials in tuning' )
    
    parser.add_argument('-t', type=int, required=True
                        , help='Do transformation 1,2 or not' )
    parser.add_argument('-f', type=str, required=True
                        , help='Data file path' )
    parser.add_argument('-lg', type=str,default='/tmp/ParkClusterAppli.log'
                        , required=False
                        , help='Log file path ' )  
                        
    parser.add_argument('-e', type=int,default=1, required=False
                        , help='Number of epochs' )
    parser.add_argument('-m', type=int,default=1, required=False
                        , help='Distance metric must be in {0,1,2,3,4}\
                        0: euclidien distance, 1: d*max(W), \n \
                        2: d*(max(W)+k) , 3: d*mean(W), 4: d*(mean(W)+k)' )
    parser.add_argument('-k', type=float,default=1, required=False
                        , help='Value of k on metric function' )
    parser.add_argument('-p', type=int,default=0, required=False
                        , help='{0,1} plot or not' )
    parser.add_argument('-l', type=int,default=20, required=False
                        , help='Log level \
                                {10: DEBUG, 20: INFO, 30: WARN, 50: FATAL' )
    return parser


# -----------------------------------------------------------------------------
#   Data
#  ----------------------------------------------------------------------------
def data():
    """
    A parameter-less function that defines and return all data needed in the above
        model definition.
    Return X_train, X_val, X_test, y_train, y_val, y_test
    """
    
    from DataTools import Data_tools
    from sklearn.model_selection import train_test_split
    
    dataUtil = Data_tools() # data tools
    
    if Cfg.DoTransformation==1:
        dataUtil.construct_transformed_data1(Cfg.datafile)
    if Cfg.DoTransformation==2:
        dataUtil.construct_transformed_data2(Cfg.datafile)
    if Cfg.DoTransformation==3:
        dataUtil.construct_transformed_data3(Cfg.datafile)
    
    X,Y = dataUtil.getDataSparse( "./" + Cfg.datafile 
                           + "_tr_" + str(Cfg.NbClusters) 
                           + "_nz_" + str(Cfg.NbZones)
                           + "_in_" + str(Cfg.InstanceNum)
                           + ".npz") # get Nbrows from data
    
    # Split data in tow sets :  train & test
    X_train,X_test,y_train,y_test = train_test_split(X, Y
                    , test_size=Cfg.pTest, random_state=Cfg.RandomState)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train
                    , test_size=Cfg.pVal, random_state=Cfg.RandomState)
    return X_train, X_val, X_test, y_train, y_val, y_test



# -----------------------------------------------------------------------------
#   main
#  ----------------------------------------------------------------------------
if __name__ == '__main__':

    print_welcome(sys.argv) # welcome
    parser = Menu() # Menu
    args = parser.parse_args() # Args 
    # _________________________________________________________________________
    # Args values
    NbIter = args.i
    Cfg.NbClusters  = args.c
    Cfg.Nbrows  = args.r
    Cfg.k = args.k
    Cfg.m = args.m
    Cfg.epochs = args.e
    Cfg.LogLevel = args.l
    Cfg.datafile = args.f
    Cfg.DoTransformation = args.t
    Cfg.RandomState = args.rs
    Cfg.NbTrials = args.nt
    Cfg.PlotHistory = args.ph
    Cfg.NbZones = args.nz
    Cfg.cm = args.cm
    Cfg.InstanceNum = args.instn
    Cfg.Normalize = args.nr
    if args.cm == 0 and  args.c > 0:
        raise ValueError( "cm = " + str(args.cm) + " and #clusters = "
                         +str(args.c) + "> 0")
    p = args.p
    logLevel = args.l 
    # _________________________________________________________________________
    # Log level
    logging.basicConfig(level=logLevel) # filename=args.lg,
    logger.setLevel(logLevel)
    
    # _________________________________________________________________________
    # random seed for reproducibility
    numpy.random.seed(Cfg.RandomState)
    
    """
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    import tensorflow as tf
    import random as rn
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    numpy.random.seed(Cfg.RandomState)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(Cfg.RandomState)
    from keras import backend as K
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(Cfg.RandomState)
    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    """
    
    
    # _________________________________________________________________________
   
    scores = [] # stock all iteration scores
    smty = [] # stock all iteration dissimilarity values
    s=0
    
    # _________________________________________________________________________
    # data 
    X_train, X_val, X_test, y_train, y_val, y_test = data()
    import sys
    
    if args.gen > 0:
        logger.info('gen-done: '+ blue(str(sys.argv)))
        logger.info("Exit after generate because -gen 1")
        sys.exit(0)
        
    Cfg.DoTransformation = False
    
    # _________________________________________________________________________
    # Parking Neural Network
    
    PNN = Parking_neural_network()
    
    dynamic = 1
    # _________________________________________________________________________
    if Cfg.dynamic==1:
        # search the best model     
        PNN.search_best_model(data=data)
    else:
        # static model :
        PNN.create_static_model(X_train)
    
    if X_test.shape[0] > 0:
        
        score = PNN.model.evaluate(X_test, y_test, verbose = 0) * 100
        scores.append(score)
        logger.info(red(" Best model [score: ") 
                                + blue( str(round(score,5))) 
                                + "]"
                                )
    
    # _________________________________________________________________________
    # add validation to train set
    if Cfg.dynamic==1:
        X_train = sparse.vstack([X_train, X_val],format="csr") 
        y_train = sparse.vstack([y_train, y_val],format="csr")
    # _________________________________________________________________________
    W = None
    Wcl = None
    lbls = None # clustering labels
    pv_centr = 'k-means++'  # kmeans center initialisation method
    
    tim = 0
    dissm_info = " "
    df_clusters = pandas.DataFrame()
    # _________________________________________________________________________
    if 0 < args.c : # if using clustering approach
        
        d_cl, cluster_names = PNN.prepare_data_for_clust(X_train)
        d_cl_t, cluster_names_t = PNN.prepare_data_for_clust(X_test)
        clust_tools = Clustering_tools(d_cl, cluster_names)
        X_train, pv_centr, lbls, X_test, lbls_t,tim = clust_tools.kmeans_v2(d_cl
                                            , d_cl_t,X_train,X_test)
        print(lbls.shape)
        df_clusters["It0"] = list(lbls[:,0])
        """print(d_cl.shape, " vs ", X_train.shape)
        df_test = pandas.DataFrame(d_cl_t.toarray(),columns=['WEATHER_EVENT'
                                    , 'HOUR_0', 'HOUR_1'
                                    , 'HOUR_2', 'HOUR_3', 'HOUR_4', 'HOUR_5', 'HOUR_6'
                                    , 'HOUR_7', 'HOUR_8', 'HOUR_9', 'HOUR_10', 'HOUR_11'
                                    , 'HOUR_12', 'HOUR_13', 'HOUR_14', 'HOUR_15', 'HOUR_16'
                                    , 'HOUR_17', 'HOUR_18', 'HOUR_19', 'HOUR_20', 'HOUR_21'
                                    , 'HOUR_22', 'HOUR_23'
                                    , 'PREVIOUS_OCCUPATION_RATE', 'HOLIDAY'])
        df_test["clusters"] = lbls_t
        parallel_coordinates(df_test, 'clusters', color=('#FFE888', '#FF9999', '#273c75') )
        plt.show()"""
        
        
    
          
    # _________________________________________________________________________
    # Training 
    from tqdm import tqdm
    values = range(NbIter)
    val_data=(X_val, y_val)
    with tqdm(total=len(values), file=sys.stdout) as pbar:  
        for iter in values:
            
            if 0 < iter :
                #====================================
                # check if W_prev is equal to W 
                W_prev = W
                W = PNN.getW()[0:X_train.shape[1],:]
                
                print("\n Max(|W|) : ",numpy.max(numpy.abs(W).max(axis=1))
                        ," argmax(|W|) : ", numpy.argmax(numpy.abs(W).max(axis=1))
                        ," Min(|W|) : ",numpy.min(numpy.abs(W).max(axis=1))
                        ," argmin(|W|) : ", numpy.argmin(numpy.abs(W).max(axis=1)))
                
                #print(numpy.abs(W).max(axis=1))
                
                
                if W is not None and W_prev is not None:
                    if ((W_prev == W).all()) :
                        logger.info(red(" Warning :   W == W_prev  "))
                #====================================            
                if Cfg.cm > 0 :
                    #====================================
                    # check if Wcl_prev is equal to Wcl 
                    Wcl_prev = Wcl
                    Wcl = PNN.getW()[Cfg.NbZones:X_train.shape[1]-Cfg.NbClusters,:]
                    if iter < NbIter:
                        if Wcl is not None and Wcl_prev is not None:
                            if ((Wcl_prev == Wcl).all()) :
                                logger.info(red(" Warning :   Wcl == Wcl_prev  "))
                        #====================================        
                        # clustering
                        pv_lab = lbls
                        
                        if Cfg.cm == 1:
                            X_train, lbls, X_test, lbls_t,tim = clust_tools.CALM(
                                            d_cl ,d_cl_t,X_train,X_test,y_train
                                            ,PNN.model,lbls,Wcl,args.k , args.m
                                            )
                        elif Cfg.cm == 2:
                            X_train, lbls, X_test, lbls_t,tim = clust_tools.Random(
                                            d_cl ,d_cl_t,X_train,X_test
                                            )
                        
                        print(lbls.shape)
                        df_clusters["It"+str(iter+1)] = list(lbls[:,0]) 
                        
                        """df_test["clusters"] = lbls_t
                        parallel_coordinates(df_test, 'clusters', color=('#FFE888', '#FF9999', '#273c75') )
                        plt.show()"""
                       
                    # dissimilarity
                    if pv_lab is not None and lbls is not None:
                        smty.append(( 100 - ((numpy.sum(pv_lab == lbls))/
                                                  d_cl.shape[0])*100))
                        dissm_info = ","+red(" di:") \
                                    + blue(str(round(smty[s],3))+"%") \
                                    + '['+red(str(tim)) +'.s]'
                        s = s+1
                    
            history,itTime = PNN.train(X_train,y_train)
            
            if Cfg.PlotHistory:
                logger.info(blue(" History.keys = ")
                                    + str(history.history.keys()))
                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.title('Model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train'], loc='upper left')
                plt.show()
            
            if X_test.shape[0] > 0:
                score = PNN.model.evaluate(X_test, y_test, verbose=0) * 100
                if score < min(scores):
                    scores.append(score)
                else:
                    scores.append(min(scores))
                    
                pbar.set_description('('+red('sc: ') 
                                    + cyan(str(round(min(scores),5)))
                                    + "% [" + blue(str((round(itTime,2))))+ '.s' +"]"
                                    + dissm_info
                                    +')'
                                      )
                    
            pbar.set_description("% [" + blue(str((round(itTime,2))))+ '.s' +"]"
                                    + dissm_info
                                    +')'
                                      )
            
            pbar.update(1)
            #sys.stdin.read(1)
    
    # _________________________________________________________________________
    # Results
    if 0 < args.c:
        df_clusters.to_csv("./clusters.csv", sep=';',index=False)
        if X_test.shape[0] > 0:
            print(str(args.m) + " , " + str(args.k)
                     + " , " + str(round(min(smty),2))
                     + " , " + str(round(max(smty),2))
                     + " , " + str(round(numpy.mean(smty),2))
                     + " , " + str(round(min(scores),5))
                     + " , " + str(round(max(scores),5)))
    # _________________________________________________________________________
    # Plotting scores results
    if X_test.shape[0] > 0:
        logger.info(" Scores : " + cyan(str(scores)))
        logger.info(" Min of scores is : " + red(str(round(min(scores),5)))
                     + " (m,k) = " + blue("(" + str(args.m) 
                     + ", " +str(args.k)+")  \n") )
        
        f = open("/tmp/scores.csv", 'w')
        #f.write(str(sys.argv)+"\n\n")
        f.write(str(scores))
        f.close()
    
    
    if p > 0 :
        
        plt.plot(scores)
        plt.xlabel('Iterations')
        plt.ylabel('Scores')
        plt.title(str(sys.argv))
        plt.show()
            
    # _________________________________________________________________________
    # Plotting dissimilarity results
    if Cfg.cm > 0:
        logger.info(blue( " Dissimilarity ")
                            +"( min: " + red(str(round(numpy.min(smty),2)))
                            + ", max: " + red(str(round(numpy.max(smty),2)))
                            + ", mean: " + red(str(round(numpy.mean(smty),2))) 
                            + " )\n ") 
        
        f = open("/tmp/diss.csv", 'w')
        f.write(str(smty))
        f.close()
        
        if p > 0 : 
            
            plt.plot(smty)
            plt.xlabel('Iterations')
            plt.ylabel('Dissimilarity %')
            plt.title(str(sys.argv))
            plt.show()
        
