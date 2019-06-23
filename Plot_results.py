#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def print_3(X,Xc,Xr,title):
    plt.plot(X,color='blue', linestyle='-')
    plt.plot(Xc,color='black', linestyle='--')
    plt.plot(Xr,color='red', linestyle=':')
    plt.legend(['Standard: '+str(round(min(X),6)), 'CALM: '+str(round(min(Xc),6)), 'Random: '+str(round(min(Xr),6))]
                , loc='upper right')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.ylabel('Errors(%)')
    plt.savefig('./ou/'+str(title)+'.pdf')
    #plt.show()
    
