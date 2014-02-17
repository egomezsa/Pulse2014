import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy import misc


def build_descriptors():
    surf = cv2.SURF(400)
    desc = list()
    desc.append([0])
    for p in range(1,151):
        string = ''.join(['PokeSprites/',str(p),'.png'])
        A = cv2.imread(string)
        A = misc.imresize(A,(200,200), 'bilinear')
        surf = cv2.SURF(400)
        kp, descriptor = surf.detectAndCompute(cv2.cvtColor(A,cv2.COLOR_BGR2GRAY),None)
        desc.append(descriptor)
    return desc
