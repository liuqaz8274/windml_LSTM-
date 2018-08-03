# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 00:00:21 2018

@author: menguan
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib as mpl  
import matplotlib.pyplot as plt  
import matplotlib.colors 
import math
import time
import csv
import sys
import os
cityname=['tehachapi','cheyenne','palmsprings','lasvegas','lancaster']
method=['KMeans','SpectralClustering','AgglomerativeClustering','Birch','DBSCAN']
def fun(citynum,methodnum,K):
    if sys.version_info < (3, ):
        mode = 'rU'
    else:
        mode = 'r'
    ed=list()
    filename='cluster/'+cityname[citynum]+method[methodnum]+str(K)
    with open(filename+'.csv', mode) as csv_arch:
                    reader = csv.reader(csv_arch, delimiter=',')
                    for row in reader:
                        ed.append({'ed':int(row[1]),'index':int(row[0])})
    
    
    ed.sort(key=lambda x:x['index'])
    data_home = "D:" + "/nrel_data/"
    archive_file_name = "meta.csv"
    archive_file = os.path.join(data_home, archive_file_name)
    clusterdata = []
    with open(archive_file, mode) as csv_arch:
        reader = csv.reader(csv_arch, delimiter=',')
        for row in reader:
            point = []
            idt=int(row[0])
            for i in range(len(ed)):
                if(ed[i]['index']==idt):
                    point.append(float(row[2]))
                    point.append(float(row[1]))
                    clusterdata.append(point)
                    break
    ans=[]
    for i in range(len(ed)):
        ans.append(ed[i]['ed'])
    
    clusterdata=np.array(clusterdata)
    cm = mpl.colors.ListedColormap(list("rgbmyc"))  
    plt.figure(figsize=(10,6),facecolor="w")    
    plt.scatter(clusterdata[:,0],clusterdata[:,1],marker='o',c=ans,s=500,cmap=cm,edgecolors="none")  
    x1_min,x2_min = np.min(clusterdata,axis=0)  
    x1_max,x2_max = np.max(clusterdata,axis=0)  
#    plt.xticks(fontsize=10)
#    plt.yticks(fontsize=10)
    plt.xlim((x1_min-0.0125,x1_max+0.0125))  
    plt.ylim((x2_min-0.01,x2_max+0.01))  
    plt.title(cityname[citynum]+' '+method[methodnum]+' K='+str(K))  
    plt.savefig(filename+'.png')
    plt.show() 
for i in range(0,5):
    
            fun(citynum=i,methodnum=4,K=0)
            