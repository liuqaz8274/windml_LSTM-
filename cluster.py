# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:20:56 2018

@author: menguan
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import time
import csv
import sys
from windml.datasets.nrel import NREL
from windml.mapping.power_mapping import PowerMapping
from windml.model.windpark import Windpark
from windml.model.turbine import Turbine
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
start = time.clock()

cityname=['tehachapi','cheyenne','palmsprings','lasvegas','lancaster']
method=['KMeans','SpectralClustering','AgglomerativeClustering','Birch','DBSCAN']


def fun(citynum,methodnum,K):
    park_id = NREL.park_id[cityname[citynum]]
    windpark = NREL().get_windpark(park_id, 10, 2004,2006)
    pla=[]
    kk=windpark.get_turbines()
    for i in range(len(kk)):
        pla.append(kk[i].idx)
        
    
    feature_window, horizon = 3, 3
    mapping = PowerMapping()   
    data_1 = np.array(mapping.get_features_park(windpark, feature_window, horizon)) 
     
    data_train = np.array(mapping.get_features_park(windpark, 1, 1))
    
    lendata=len(data_1)
    data1 = data_1[:lendata:3]
    l1=len(data_train)
    data_train1=data_train[:l1:3]
    half=int(math.floor(len(data1) * 0.5))
    
    traindata_1=data_train1[0:half,:]
    traindata1=np.transpose(traindata_1)
    traindata1=preprocessing.scale(np.array(traindata1),with_mean=True,with_std=True)
    if methodnum==0:
        ans = KMeans(n_clusters=K, random_state=0).fit(traindata1).predict(traindata1)
    if methodnum==1:
        ans = SpectralClustering(n_clusters=K, random_state=0).fit_predict(traindata1)
    if methodnum==2:
        ans = AgglomerativeClustering(n_clusters=K).fit_predict(traindata1)
    if methodnum==3:
        ans = Birch(n_clusters=K).fit_predict(traindata1)
    if methodnum==4:
        ans = DBSCAN(eps = 0.1).fit_predict(traindata1)
    fo = open('cluster10/'+cityname[citynum]+method[methodnum]+str(K)+'.csv','w', newline='')
    csv_write = csv.writer(fo,dialect='excel')
    for i in range(len(ans)):
        cc=[];
        cc.append(pla[i])
        cc.append(ans[i])
        csv_write.writerow(cc)
    fo.close()
for i in range(0,5):
    for j in range(0,4):
            fun(citynum=i,methodnum=j,K=3)
            fun(citynum=i,methodnum=j,K=2)
#for i in range(0,5):
#    fun(citynum=i,methodnum=4,K=0)
    