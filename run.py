# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 01:17:04 2018

@author: menguan
"""

import os
for i in range(0,1):
    for j in range(0,4):
            os.system("python test2.py "+str(i)+' '+ str(j)+" 5")
            os.system("python test2.py "+str(i)+' '+ str(j)+" 7")
            os.system("python test2.py "+str(i)+' '+ str(j)+" 9")
            os.system("python test2.py "+str(i)+' '+ str(j)+" 11")
