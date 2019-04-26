"""
queryEval.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 


@author: John See, 2017
@modified by: Lai Kuan, Wong, 2018

"""
import os
import cv2
import numpy as np
import pickle
import sys, getopt
import matplotlib.pyplot as plt
from computeFeatures import computeFeatures
from computeDistances import computeDistances

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'D:\\Document\\MMU Sem 6\\Visual Information Processing\\Assignment\\Assignment 2\\plantdb\\train'

# these labels are the classes assigned to the actual plant names
labels = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"d:q:h")

# parsing command line args
for o, a in myopts:
    print(a)
    if o == '-d':
        queryfile = os.path.join(dbpath, a + '.jpg')
        gt_idx = np.uint8(np.floor((int(a))/50)) 
        print()
        if not os.path.isfile(queryfile):
            print("Error: Query file does not exist! Please check.")
            sys.exit()
    elif o == '-q':
        queryfile = a
        if not os.path.isfile(queryfile):
            print("Error: Query file does not exist! Please check.")
            sys.exit()
        # tokenize filename to get category label and index
        gt = str(queryfile.split("_")[1]).split(".")[0]
        gt_idx = labels.index(gt)
    elif o == '-h':
        print("\nUsage: %s -d dbfilenumber\n       # to specify a single query image from the database for evaluation" % sys.argv[0])
        print("\n       %s -q queryfile\n       # to specify a new query image for evaluation" % sys.argv[0])
        print(" ")       
        sys.exit()
    else:
        print(' ')
    

featvect = []  # empty list for holding features
FEtime = np.zeros(500)

# load pickled features
fv = pickle.load(open("feat.pkl", "rb") )
print('Features loaded')

# read query image file
img = cv2.imread(queryfile)
query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# show stuff
plt.imshow(query_img), plt.title('Query image: %s'%labels[gt_idx])
plt.xticks([]), plt.yticks([])
print('Query image: %s'%labels[gt_idx])

# compute features
newfeat = computeFeatures(query_img)


# insert new feat to the top of the feature vector stack
fv = np.insert(fv, 0, newfeat, axis=0)

# find all pairwise distances
D = computeDistances(fv)


# *** Evaluation ----------------------------------------------------------

# number of images to retrieve
nRetrieved = 10

# access distances of all images from query image (first image), sort them asc
nearest_idx = np.argsort(D[0, :]);

# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor((nearest_idx[1:nRetrieved+1])/50));
 
# find matches
hits_q = (retrievedCats == gt_idx)
  
# calculate average precision of the ranked matches
if np.sum(hits_q) != 0:
  avg_prec_q = np.sum(hits_q*np.cumsum(hits_q)/(np.arange(nRetrieved)+1)) / np.sum(hits_q)
else:
  avg_prec_q = 0.0
          
recall = np.sum(hits_q) / nRetrieved

# *** Results & Visualization-----------------------------------------------

print('Average Precision, AP@%d: %.4f'%(nRetrieved,avg_prec_q))
print('Recall Rate@%d: %.4f'%(nRetrieved,recall)) 

fig, axs = plt.subplots(2, 5, figsize=(15, 6), facecolor='w', edgecolor='w')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(10):
    imgfile = os.path.join(dbpath, str(nearest_idx[i+1]) + '.jpg')
    matched_img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
    axs[i].imshow(matched_img)
    axs[i].set_title(str(i+1) + '. ' + str(labels[retrievedCats[i]]))
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    
plt.show()