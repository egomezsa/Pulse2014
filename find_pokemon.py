import prep
import numpy as np
import scipy.spatial.distance as sp
import scipy.stats.mstats as mstats
import cv2
import matplotlib.pyplot as plt
import sys
from sklearn import decomposition

def test_frame(frame,desc):
    surf = cv2.SURF(400)

    #kp, descriptor = surf.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY), None)
    kp, descriptor = surf.detectAndCompute(frame, None)
   
    if descriptor is None:
        return -1 
    points = descriptor.shape[0]
    matches = np.zeros((points,1))
    best_poke_desc = np.zeros((points,151))

    best_poke_desc[:,0] = 151;

    for pt in range(0, points):
        d = descriptor[pt,:].reshape(1,128)
        for poke in range(1,151):
            best_poke_desc[pt,poke] =  np.min(sp.cdist(desc[poke],d,'Euclidean'))
        matches[pt] = np.argmin(best_poke_desc[pt,:])
    val, count =  mstats.mode(matches)
    if( count[0][0] < 4):
        return -1

    return val[0][0]
#return mstats.mode(matches)[0][0][0]


def compare_image(file_name, window):
    desc = prep.build_descriptors()
    A = cv2.imread(file_name)
    results = list()
    frame = np.zeros((window,window,3))
    for row in range(0, A.shape[0], window/5):
        results.append('r')
        for col in range(0, A.shape[1], window/5):
            frame = A[row:row+window, col:col+window,:]
            if frame.shape[0]  != window or frame.shape[1] != window:
                break
            results.append(test_frame(frame,desc)) 
    return results
    
    

def build_descriptors():
    surf = cv2.SURF(400)
    desc = list()
    desc.append([0])
    for p in range(1,151):
        string = ''.join(['PokeSprites/', str(p),'.png'])
        A = cv2.imread(string)
        surf = cv2.SURF(400)
        kp, descriptor = surf.detectAndCompute(cv2.cvtColor(A,cv2.COLOR_BGR2GRAY),None)
        desc.append(descriptor)
    return desc
