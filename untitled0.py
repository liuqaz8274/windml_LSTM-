import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import numpy as np
import scipy
import numpy as np
from PIL import Image
import matplotlib
import csv 
import cv2


imgs = np.load('imgs.npy')
label = np.load('labels.npy')

#imgs=imgs.reshape((20,224,224,3))
#imgs=imgs*255.0

#for i in range(0,20):
#    print("11")     
#    cv2.imwrite('C:/Users/menguan/Desktop/代码/appmodels/imgs/test'+str(i)+'.png', np.zeros((10,10)))
#    scipy.misc.imsave('C:/Users/menguan/Desktop/代码/appmodels/imgs/test'+str(i)+'.png', imgs[i])
#    matplotlib.image.imsave('C:/Users/menguan/Desktop/代码/appmodels/imgs/test'+str(i)+'.jpg', imgs[i])
    
im = Image.open('C:/Users/menguan/Desktop/代码/appmodels/imgs/test5.png')
im.resize([224,224], Image.ANTIALIAS)
im2 = np.array(im)

im2 = im2.reshape(-1)
#fo = open('test0.csv','a', newline='')
#csv_write = csv.writer(fo,dialect='excel')
#csv_write.writerow(imgs[0])
#fo.close()