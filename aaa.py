# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:36:30 2018

@author: menguan
"""

from __future__ import print_function
import numpy as np
import time
import csv
import sys
from windml.datasets.nrel import NREL
from windml.mapping.power_mapping import PowerMapping
from windml.model.windpark import Windpark
from windml.model.turbine import Turbine


start = time.clock()
def down(citynum, methodnum):
    cityname=['tehachapi','cheyenne','palmsprings','lasvegas','lancaster']
    method=['KMeans','SpectralClustering','AgglomerativeClustering','Birch']
    target_idx = NREL.park_id[cityname[citynum]]
    year_from,year_to=2004,2006
    if sys.version_info < (3, ):
        mode = 'rU'
    else:
        mode = 'r'
    csvfile='cluster/'+method[methodnum]+'200.csv'
    pick=[]
    with open(csvfile, mode) as csv_arch:
                reader = csv.reader(csv_arch, delimiter=',')
                for row in reader:
                    pick.append(int(row[0]))

    nrel=NREL()
    d=0
    turbines = nrel.fetch_nrel_meta_data_all()

    for row in turbines:
            turbine_index = np.int(row[0])
            if (turbine_index != target_idx):
                if(pick[turbine_index-1]==pick[target_idx-1]):
                    d=d+1
                    for y in range(year_from, year_to+1):
                           measurement = nrel.fetch_nrel_data(row[0], y, 
                                                              ['date','corrected_score','speed'])
for x in range(0, 5):      
    for y in range(0,4):                     
        down(x,y)
elapsed = (time.clock() - start)
print("Time used:",elapsed) 