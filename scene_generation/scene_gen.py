#!/usr/bin/python
import random
import numpy as np
import cv2
import scipy.misc as sm
import csv
import sys

def unique_random(choices):
    while True:
        r = random.randrange(len(choices) - 1) + 1
        choices[0], choices[r] = choices[r], choices[0]
        yield choices[0]

def overlayImage(src, pke, y_loc, x_loc, im_size):
    if y_loc + im_size > src.shape[0] or x_loc + im_size > src.shape[1]:
        im_size = min(src.shape[0] - y_loc, src.shape[1] - x_loc)
    for c in range(0,3):
        src[y_loc: y_loc + im_size, x_loc : x_loc + im_size, c] = pke[0:im_size,0:im_size,c] * (pke[0:im_size,0:im_size,3]/255.0) +  src[y_loc : y_loc + im_size, x_loc : x_loc + im_size, c] * (1.0 - pke[0:im_size,0:im_size,3]/255.0)
    return src

def generate_occlusion(allowOcclusion, background, scene, scale, y_val, x_val):
    hide = random.randint(1,3)
    if allowOcclusion == True and hide == 1:
        occ_str = ''.join(['occlusion/',str(scene),'.png'])
        occlusion = cv2.imread(occ_str,-1)
        occlusion = sm.imresize(occlusion,(80, 80))
        occ_y = y_val + 80
        occ_x = x_val + 40
        background = overlayImage(background,occlusion,occ_y, occ_x,occlusion.shape[0])
    return background

def load_data():
    poke_info = dict()
    scene_info = dict()
    with open('image_data.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            index = int(row[0])
            scene = int(row[1])
            scale = float(row[2])
            poke_info[index] = (scene, scale)  
            if scene not in scene_info:
                scene_info[scene] = list()
            else:
                scene_info[scene].append(index)




    return poke_info, scene_info

def generateImage(numb_pokemon, allowOcclusion, allowScale, basicTest):

    random.seed()
    if numb_pokemon > 5:
        numb_pokemon = 5    
    poke_info, scene_info = load_data()      

    poke_list = list()
    poke_def_size = 150
    #Setting a random seed pokemon to determine the seed. 
    
    seed_index = random.randint(1,151)
    scene =  poke_info[seed_index][0]
    scene_val = scene - 1
    scene_file = scene
    if scene == 5:
        scene_file = random.choice([2,4])
        scene_val = random.choice([5,6])
    if scene == 3:
        scene_val = random.choice([2,3])
    scene_name = ''.join(['',str(scene_file),'.png'])

    #Setting up the background 

    background = cv2.imread(scene_name)
    background = sm.imresize(background, (680,1080))
    background = cv2.GaussianBlur(background,(11,11),0)
    ranges = np.array([[0,310,880,390],[0,310,880,390], [0,310,880,390],[0,310,880,390],[0,0,780,170],[0,0,780,160],[0,0,780,230]])
    if basicTest == True:
        background[:,:,0] = 0;
        background[:,:,1] = 0;
        background[:,:,2] = 255;
        numb_pokemon = 1
        allowOcclusion = False
        allowScale = False
    poke_str = ''.join(['PokeSprites/',str(seed_index),'.png'])
    poke_list.append(seed_index)
    p = cv2.imread(poke_str, -1)
    p =  sm.imresize(p,(poke_def_size,poke_def_size))
    if allowScale == True:
        scale = poke_info[seed_index][1]
        p = sm.imresize(p,1.0)
    
    x_val = np.random.random_integers(ranges[scene_val][0],ranges[scene_val][2], (1, numb_pokemon))[0]
    y_val = np.random.random_integers(ranges[scene_val][1],ranges[scene_val][3], (1, numb_pokemon))[0]
    
    w_step = 880 / numb_pokemon
    horizontal = np.zeros((1,numb_pokemon))
    for r in range(0,numb_pokemon):
        horizontal[0][r] = w_step * r

    x_val =  np.random.random_integers(0,200,numb_pokemon) + horizontal
    x_val = x_val[0]

    background = overlayImage(background,p,y_val[0],x_val[0],p.shape[0])
    
    background = generate_occlusion(allowOcclusion, background,scene,p.shape[0]/2, y_val[0], x_val[0])

   
    #print scene_info[scene]
    for p_index in range(1,numb_pokemon):
        p_generated = random.choice(scene_info[scene])
        poke_list.append(p_generated)
        poke_str = ''.join(['PokeSprites/',str(p_generated),'.png'])
        p = cv2.imread(poke_str, -1)
        p = sm.imresize(p,(poke_def_size,poke_def_size))
        if allowScale == True:
            scale = poke_info[seed_index][1]
            scale = 1.5
            p = sm.imresize(p,scale)
        background = overlayImage(background,p,y_val[p_index],x_val[p_index],p.shape[0])
        background = generate_occlusion(allowOcclusion, background,scene,p.shape[1]/2, y_val[p_index], x_val[p_index])
        
    return (background, poke_list)

# Forest 1
# Ocean 2
# Mountain 3
# Grassland 4
#(im, poke_list) = generateImage(int(sys.argv[1]), int(sys.argv[2]) == 1, int(sys.argv[3]) == 1, sys.argv[4] == 'test')

for m in range(0,20):
    if m < 10:
        (im, poke_list) = generateImage(int(m/5)+1, 1, 0, 'test')
    else:
        (im, poke_list) = generateImage(int(m/5)+1, 1, 0, 'none')
    cv2.imwrite(''.join(['out',str(m),'.png']), im)
    print poke_list
