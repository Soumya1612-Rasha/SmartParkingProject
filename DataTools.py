#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import pandas
import numpy
import random
from scipy import sparse # import sparse module from SciPy package 
from time import perf_counter
from colors import red, green, cyan, blue, yellow
from Configuration import Cfg

# -----------------------------------------------------------------------------
#  Transform_one_row1
# -----------------------------------------------------------------------------
def transform_one_row1(arg):
    """ 
    Get tranformed row of data  with approach 1
    output X_row: sparse row matrix
    """
    idx,row,columns,nbclust,Weather_events = arg
    X_row = pandas.DataFrame(numpy.zeros((1, len(columns))),
                     columns=columns)
    
    X_row.at[0, 'OCCUPATION_RATE'] = row['OCCUPATION_RATE']
    X_row.at[0, "WEEKDAY_" + str(row['START_WEEKDAY'])] = 1
    X_row.at[0,  "HOUR_" + str(row['START_HOUR'])] = 1
    X_row.at[0,  row['BLOCK_ID']] = 1
    
    if (   row['PREVIOUS_OCCUPATION'] is None
        or row['PREVIOUS_OCCUPATION'] == 'None'):
        
        X_row.at[0, 'PREVIOUS_OCCUPATION_RATE'] = row['OCCUPATION_RATE']
    else:
        X_row.at[0, 'PREVIOUS_OCCUPATION_RATE'] = row['PREVIOUS_OCCUPATION']
        
    X_row.at[0, 'SPOTS_NUMBER'] = row['SPOTS_NUMBER']
    X_row.at[0, 'TEMPERATURE'] = row['TEMPERATURE']
    
    if not (pandas.isnull(row['WEATHER_EVENT']) or 
            row['WEATHER_EVENT'] in (' ', 'ND','None')):
        if row['WEATHER_EVENT'] in Weather_events:
            X_row.at[0,  row['WEATHER_EVENT']] = 1
            
            
    if not (pandas.isnull(row['HOLIDAY']) or
            row['HOLIDAY'] in (' ', 'ND','None')):
        X_row.at[0,  'HOLIDAY'] = 1
        
    if not (pandas.isnull(row['BEFORE_HOLIDAY']) or 
            row['BEFORE_HOLIDAY'] in (' ', 'ND','None')):
        X_row.at[0,  'BEFORE_HOLIDAY'] = 1
        
    if 0 < nbclust:  
        cluster = random.randint(1,nbclust)
        X_row.at[0,  'CLUSTER_'+str(cluster)] = 1
        
    return sparse.csr_matrix(X_row)
# -----------------------------------------------------------------------------
#  Transform_one_row2
# -----------------------------------------------------------------------------
def transform_one_row2(arg):
    """ 
    Get tranformed row of data  with approach 2
    output X_row: sparse row matrix
    """
    idx,row,columns,nbclust = arg
    X_row = pandas.DataFrame(numpy.zeros((1, len(columns))),
                     columns=columns)
    #print(columns)
    #print(idx)
    X_row.at[0, "HOUR_" + str(row['START_HOUR'])] = 1
    X_row.at[0, row['BLOCK_ID']] = 1
    
    
    if (   row['PREVIOUS_OCCUPATION'] is None
        or row['PREVIOUS_OCCUPATION'] == 'None'):
        X_row.at[0,'PREVIOUS_OCCUPATION_RATE'] = row['OCCUPATION_RATE']
    else:
        X_row.at[0,'PREVIOUS_OCCUPATION_RATE'] = row['PREVIOUS_OCCUPATION']
    
    X_row.at[0, 'SPOTS_NUMBER'] = row['SPOTS_NUMBER']
    X_row.at[0,'OCCUPATION_RATE'] = row['OCCUPATION_RATE']
    
    if (row['WEATHER_EVENT'] is not None) :
        if row['WEATHER_EVENT'] in ['Rain','Haze','Mist','Fog','Drizzle']:       
            X_row.at[0, 'WEATHER_EVENT'] = 1
        
    if not (pandas.isnull(row['HOLIDAY']) or
            row['HOLIDAY'] in (' ', 'ND','None')) or (
                    row['START_WEEKDAY'] == 6 
                    or row['START_WEEKDAY'] == 7):
        X_row.at[0, 'HOLIDAY'] = 1
        
    if not (pandas.isnull(row['BEFORE_HOLIDAY']) or 
            row['BEFORE_HOLIDAY'] in (' ', 'ND','None')):
        X_row.at[0,  'HOLIDAY'] = 1
        
    if 0 < nbclust:  
        cluster = random.randint(1,nbclust)
        X_row.at[0, 'CLUSTER_'+str(cluster)] = 1
    
    
    #print(X_row)
    return sparse.csr_matrix(X_row)

# -----------------------------------------------------------------------------
# Clustering_tools
# -----------------------------------------------------------------------------
class Data_tools(object):
    """
    Data tools for parking 
    """
    def __init__(self):
        """
        Constructor :
            param :
                string file : path of input file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
    # _________________________________________________________________________
    def getData(self,file):
        """ 
        Get Data From CSV File
        output df : dataframe
        """
        self.logger.info(' Load in data from CSV file "'+ str(file)+'"')
        t0 = perf_counter() # timer 
        
        X = pandas.read_csv(file, sep=';'
                             , encoding="ISO-8859-1"
                             , nrows = Cfg.Nbrows)
        
        X.drop(X.columns[X.columns.str.contains('unnamed',case = False)]
                            ,axis=1, inplace =True)
        
        self.logger.info(' Loading data is finished with success.')
        self.logger.info(green(' Time taken  : ') + red(str(perf_counter()-t0))
                            + " " +green("Sec."))
                            
        self.logger.info(str(X.head(2)))  # Display df
        self.logger.info( " Before drop_duplicates : " + red(str(X.shape)))
        #X.drop_duplicates( keep='first', inplace=True)
        """df=df.reset_index().drop_duplicates(subset=['OCCUPATION_RATE'
                                   , 'PREVIOUS_OCCUPATION'
                                   , 'SPOTS_NUMBER'
                                   ,'TEMPERATURE'
                                   , 'WEATHER_EVENT'
                                   , 'HOLIDAY'
                                   , 'BEFORE_HOLIDAY'
                                   , 'SPECIAL_EVENT']
                                   ,keep='first').set_index('index')"""
        
        self.logger.info( " After drop_duplicates : "+ cyan(str(X.shape)))
        
        Y = X['OCCUPATION_RATE']
        X.drop(['OCCUPATION_RATE'], axis=1, inplace =True)
        return  X,Y
    
    # _________________________________________________________________________
    def getDataSparse(self,file):
        """ 
        Get Data From .npz File
        output  X,Y: sparse row matrix 
        """
        from sklearn import preprocessing
        self.logger.info(' Load in data from .npz file "'+ str(file)+'"')
        t0 = perf_counter() # timer 
        X = sparse.load_npz(file)
        self.logger.info(' Loading data is finished with success.')
        self.logger.info(green(' Time taken  : ') 
                            + red(str(round(perf_counter()-t0,5)))
                            + " " +green("Sec.")
                            + ",  X.shape ="
                            + red(str(X.shape)))
        
        
        print(" +++++++++  ", X.shape)
        print(X[0,:].todense())
        
        Y = X[:,X.shape[1]-1]
        X = X[:,0:(X.shape[1]-1)]
        
        # input normalization
        if Cfg.Normalize:
            X = preprocessing.normalize(X,axis=0, norm='l2')
            self.logger.info('Using data normalization...')
        return  X,Y


    # _________________________________________________________________________
    def construct_transformed_data1(self,file):
        """ 
        Get Data From CSV File, transform data, save data 
        input file: string, path to file
        input nb_clust: int, number of clusters
        output df: dataframe
        """
        self.logger.info(' Load in origin data from CSV file "'+ str(file)+'"')
        t0 = perf_counter() # timer 
        df = pandas.read_csv(file, sep=';'
                             , encoding="ISO-8859-1"
                             , nrows = Cfg.Nbrows)
        #df = df[df["SPOTS_NUMBER"] < 4]
        
        if Cfg.NbZones > len(df["BLOCK_ID"].unique()) :
            message = str(Cfg.NbZones) +" > "\
                                + str(len(df["BLOCK_ID"].unique()))\
                                +" zones"
            raise ValueError(message)
        #==================================
        self.logger.info(" " + red(str(Cfg.NbZones)) +"/"
                                + blue(str(len(df["BLOCK_ID"].unique())))\
                                +" .zones.")
        zones = []   
        for i in range(0,Cfg.InstanceNum):
          zones = numpy.random.choice(df["BLOCK_ID"].unique()
                              , Cfg.NbZones
                              , replace=False)         
        df = df.loc[df["BLOCK_ID"].isin(zones)]
        self.logger.info('Selected BLOCK_IDs are :'+blue(str(zones)))
        
        self.logger.info(' Loading data(ISO-8859-1) took ' 
                            + red(str(perf_counter()-t0))
                            + " " +green("Sec."))
        #==================================
        self.columns = []
        # Block Ids column
        self.Block_list = df["BLOCK_ID"].unique()
        self.logger.debug(cyan(" BLOCK IDS LIST : \n") 
                                + str(self.Block_list))
        for block in self.Block_list:
            self.columns.append(block)
            
        # weather events column
        self.Weather_events = df["WEATHER_EVENT"].unique()
        self.logger.debug(cyan(" WEATHER_EVENT list : \n") 
                                + str(self.Weather_events))
        
        for event in self.Weather_events:
            self.columns.append(event)
        
        # Week Days column
        for i in range(1, 8):
            self.columns.append("WEEKDAY_" + str(i))
        # Hour column
        for j in range(0, 24):
            self.columns.append("HOUR_" + str(j))
        
        # Previous occupation rate column
        self.columns.append('PREVIOUS_OCCUPATION_RATE')
        
        
        # Temperature column
        self.columns.append('TEMPERATURE')
        # Holiday  column
        self.columns.append('HOLIDAY')
        # Before Holiday column
        self.columns.append('BEFORE_HOLIDAY')
        
        
        # Clusters
        self.nb_clust = Cfg.NbClusters 
        
        if 0 < self.nb_clust :
            
            self.logger.debug(cyan(" Number of clusters: ") 
                                + str(self.nb_clust))
            
            for cl in range(1,self.nb_clust+1):
                self.columns.append('CLUSTER_'+ str(cl))
        else :
            self.logger.info(blue(" Neural Network without clusters.") )
            
        # Occupation rate column
        self.columns.append('OCCUPATION_RATE')
        
        self.logger.info(' Transform 1 of input is started .... ')
        t0 = perf_counter() # timer 
        
        # Do transformation for each row in parallel Mode 
        import multiprocessing as mp
        self.logger.info(" Le nombre de cpus : "+red(str(mp.cpu_count()-1)))
        pool = mp.Pool(processes=(mp.cpu_count()-1))
        tr_rows = pool.map( transform_one_row1, 
                            [(idx,row,self.columns,self.nb_clust
                                  ,self.Weather_events) 
                                for idx,row in df.iterrows()])
        X = sparse.vstack( tr_rows )
        pool.close()
        pool.join()
        # save data
        sparse.save_npz("./" +Cfg.datafile 
                           +"_tr_"+str(Cfg.NbClusters)
                           + "_nz_"  +str(Cfg.NbZones)
                           + "_in_" + str(Cfg.InstanceNum)
                           +".npz", X)
        self.logger.info(' The input data is ready in : ' 
                         +  red(str(perf_counter()-t0)) + " " + green("Sec.") )
        self.logger.info(cyan(' The input data dimensions is : ') + str(X.shape))
        print(" Features are: ", self.columns)
        
    # _________________________________________________________________________
    def construct_transformed_data2(self,file):
        """ 
        Get Data From CSV File, transform data, save data 
        input file: string, path to file
        input nb_clust: int, number of clusters
        output df: dataframe
        """
        self.logger.info(' Load in origin data from CSV file "'+ str(file)+'"')
        t0 = perf_counter() # timer 
        df = pandas.read_csv(file, sep=';'
                             , encoding="ISO-8859-1"
                             , nrows = Cfg.Nbrows)
        #================================ 
        # zones filtre
        filtre = [ 20201, 20405, 20506, 21629, 21732, 21733, 21934
                  , 32504, 33601, 33602, 33901, 33902, 33903, 33904
                  , 35902, 37001, 41100, 41101, 41102, 41103, 41322
                  , 41531, 41532, 44305, 44602, 44603, 44721, 44722
                  , 46300, 46402, 47000, 47001, 47006, 47106, 47300
                  , 50024, 50226, 50227, 50228, 52004, 56801, 56820
                  , 56821, 56822, 56823, 56824, 56825, 56826, 56827
                  , 58200, 58501, 58502, 58506, 60602, 65800, 65801
                  , 65802, 65804, 66100, 66423, 67100, 67101, 68126
                  , 68127, 82606, 86404 ]
        
        self.logger.info(" Filtre of zones contain: "+ cyan(str(filtre)))
        df = df.loc[~df["BLOCK_ID"].isin(filtre)]
        
        #================================ 
        
        if Cfg.NbZones > len(df["BLOCK_ID"].unique()) :
            message = str(Cfg.NbZones) +" > "\
                                + str(len(df["BLOCK_ID"].unique()))\
                                +" zones"
            raise ValueError(message)
        #================================ 
        
        self.logger.info(" " + red(str(Cfg.NbZones)) +"/"
                                + blue(str(len(df["BLOCK_ID"].unique())))\
                                +" .zones.")
        """
        #================================ 
        # statistics
        pmax = []
        for blockid in df["BLOCK_ID"].unique():
            dfbid = df.loc[df["BLOCK_ID"] == blockid]
            pmax.append(dfbid.shape[0])
            print("blockid = ", blockid
                  , " : #spots()= "
                  , dfbid["SPOTS_NUMBER"].unique()
                  , " we have  "
                  
                  , dfbid.shape[0]
                  , " samples ( "
                  , dfbid.loc[dfbid["OCCUPATION_RATE"]  <=  0.10].shape[0]
                  
                  ," is '<= 0.10' and "
                  , dfbid.loc[dfbid["OCCUPATION_RATE"]  == 1.0].shape[0]
                  ,"  '==1')"
                  
                  
                  )
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")           
        print("Block Id : ",df["BLOCK_ID"].unique()[numpy.argmax(pmax)])
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")          
        print( " SPECIAL_EVENT : ",df["SPECIAL_EVENT"].unique())
        
        print( " WEATHER_EVENT: ",df["WEATHER_EVENT"].unique())"""
        #================================
        
        
        
         # zones selection
        zones = []   
        for i in range(0,Cfg.InstanceNum):
          zones = numpy.random.choice(df["BLOCK_ID"].unique()
                              , Cfg.NbZones
                              , replace=False) 
        
       
        df = df.loc[df["BLOCK_ID"].isin(zones)]
        self.logger.info('Selected BLOCK_IDs are :'+blue(str(zones)))
        

        print( " SPECIAL_EVENT : ",df["SPECIAL_EVENT"].unique())
        print( " WEATHER_EVENT: ",df["WEATHER_EVENT"].unique())
        #================================
        
        self.logger.info(' Loading data(ISO-8859-1) took ' 
                            + red(str(perf_counter()-t0))
                            + " " +green("Sec. ") + "shape="+ str(df.shape))
        
        self.columns = []
        # Block Ids column
        self.Block_list = df["BLOCK_ID"].unique()
        self.logger.debug(cyan(" BLOCK IDS LIST : \n") 
                                + str(self.Block_list))
        for block in self.Block_list:
            self.columns.append(block)
            
        # weather events column
        self.columns.append("WEATHER_EVENT")
       
        
        # Hour column
        for j in range(0, 24):
           self.columns.append("HOUR_" + str(j))
        
        # Previous occupation rate column
        self.columns.append('PREVIOUS_OCCUPATION_RATE')
        
        
        # Holiday  column
        self.columns.append('HOLIDAY')
        # SPOTS_NUMBER
        self.columns.append('SPOTS_NUMBER')
        
        # Special event
        # self.columns.append('SPECIAL_EVENT')
        # Clusters
        self.nb_clust = Cfg.NbClusters 
        
        if 0 < self.nb_clust :
            
            self.logger.debug(cyan(" Number of clusters: ") 
                                + str(self.nb_clust))
            
            for cl in range(1,self.nb_clust+1):
                self.columns.append('CLUSTER_'+ str(cl))
        else :
            self.logger.info(blue(" Neural Network without clusters.") )
        
        # Occupation rate column
        self.columns.append('OCCUPATION_RATE')
        
        
        self.logger.info(' Transform 2 of input is started .... ')
        t0 = perf_counter() # timer 
        
        # Do transformation for each row in parallel Mode 
        import multiprocessing as mp
        self.logger.info(" Le nombre de cpus : "+red(str(mp.cpu_count()-1)))
        pool = mp.Pool(processes=(mp.cpu_count()-1))
        tr_rows = pool.map( transform_one_row2, 
                            [(idx,row,self.columns,self.nb_clust) 
                                for idx,row in df.iterrows()])
        X = sparse.vstack( tr_rows )
        pool.close()
        pool.join()
        
        # save data
        sparse.save_npz("./" +Cfg.datafile 
                           +"_tr_"+ str(Cfg.NbClusters)
                           + "_nz_"+ str(Cfg.NbZones)
                           + "_in_" + str(Cfg.InstanceNum)
                           +".npz", X)
        self.logger.info(' The input data is ready in : ' 
                         +  red(str(perf_counter()-t0)) + " " + green("Sec.") )
        self.logger.info(cyan(' The input data dimensions is : ') + str(X.shape))
        self.logger.info(yellow(" Features are: "+ str(self.columns)))
   
