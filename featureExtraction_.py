"""
featureExtraction.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: John See, 2017
@modified by: Lai Kuan, Wong, 2018

"""
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from computeFeatures import computeFeatures

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'D:\\Document\\MMU Sem 6\\Visual Information Processing\\Assignment\\Assignment 2\\plantdb\\train\\'

# these labels are the classes assigned to the actual plant names
labels = (1,2,3,4,5,6,7,8,9,10)
    
featvect = []  # empty list for holding features
FEtime = np.zeros(500)

for idx in range(500):
    img = cv2.imread( os.path.join(dbpath, str(idx+1) + ".jpg") )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # display image
    #plt.imshow(img), plt.xticks([]), plt.yticks([])
    #plt.show()
    
    # compute features and append to list
    e1 = cv2.getTickCount() # start timer
    feat = computeFeatures(img)
    e2 = cv2.getTickCount()  # stop timer
    
    featvect.append( feat ); 
    FEtime[idx] = (e2 - e1) / cv2.getTickFrequency() 
    
    print('Extracting features for image #%d'%idx )

print('Feature extraction runtime: %.4f seconds'%np.sum(FEtime))

temparr = np.array(featvect)
# fv = np.array(featvect)
# print(fv.shape)
print(temparr.shape)
fv = np.reshape(temparr, (temparr.shape[0], temparr.shape[1]) )
print(fv.shape)
del temparr

# pickle your features
pickle.dump( fv, open( "feat.pkl", "wb" ) )
print('Features pickled!')
