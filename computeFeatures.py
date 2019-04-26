"""
computeFeatures.py

YOUR WORKING FUNCTION for computing features

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from scipy.cluster.vq import kmeans, kmeans2, vq

# you are allowed to import other Python packages above
##########################
def computeFeatures(img):
    # Inputs
    # img: 3-D numpy array of an RGB color image
    #
    # Output
    # featvect: A D-dimensional vector of the input image 'img'
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    

    # img = cv2.imread(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #print(gray.shape)
    
    # extract SIFT features
    _sift = cv2.xfeatures2d.SIFT_create(nfeatures = 10,contrastThreshold=0.1)
    _kps, featvect = _sift.detectAndCompute(gray, None) #extracting keypoint and descriptor of the query image 
    print(featvect.shape)
    
    # load the pickled codebook, apply vq to the features
# =============================================================================
#     codebook = pickle.load( open( "feat.pkl", "rb" ) )
#     featvect, distortion = vq(_des, codebook)
# =============================================================================
    
    
    # plot
    # bowhist = np.histogram(_code, k, normed=True)
    # BOW histogram for image
    # plt.bar(bowhist[1][1:], bowhist[0]);  #bow[i][0]=>array, bow[i][1]=>bin_edges
    # plt.show()
# =============================================================================
#     plt.figure(figsize=(8,6))
#     plt.imshow(img)
#     fig = plt.gcf()
#     ax = fig.gca()
#     for r in np.arange(len(_kps)): 
#         circle1 = plt.Circle((_kps[r].pt[0], _kps[r].pt[1]), _kps[r].size/2, color='r', fill=False)    
#         ax.add_artist(circle1)
#     plt.show()
# =============================================================================
    
    
    
    # This is the baseline method: 192-D RGB colour feature histogram
    rhist, rbins = np.histogram(img[:,:,0], 64, normed=True)
    ghist, gbins = np.histogram(img[:,:,1], 64, normed=True)
    bhist, bbins = np.histogram(img[:,:,2], 64, normed=True)
    featvect3 = np.concatenate((rhist, ghist, bhist))
    print(featvect3.shape)
    
    # This creates a 300-D vector of random values as features!     
    featvect2 = np.random.rand(300,1)
    # print(featvect2.shape)
    #print(featvect)
    
    # END OF YOUR CODE
    #########################################################################
    return featvect[0][:300]