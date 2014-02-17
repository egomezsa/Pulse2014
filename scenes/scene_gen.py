import random
import numpy as np
import cv2
import scipy.misc as sm
import csv


def unique_random(choices):
    while True:
        r = random.randrange(len(choices) - 1) + 1
        choices[0], choices[r] = choices[r], choices[0]
        yield choices[0]

def overlayImage(src, pke, y_loc, x_loc, im_size):
    for c in range(0,3):
        src[y_loc: y_loc + im_size, x_loc : x_loc + im_size, c] = pke[:,:,c] * (pke[:,:,3]/255.0) +  src[y_loc : y_loc + im_size, x_loc : x_loc + im_size, c] * (1.0 - pke[:,:,3]/255.0)
    return src

def generate_occlusion(allowOcclusion, background, scene, scale, y_val, x_val):
    hide = random.randint(1,3)
    if allowOcclusion == 1 and hide == 1:
        occ_str = ''.join(['scenes/occlusion/',str(scene),'.png'])
        occlusion = cv2.imread(occ_str,-1)
        occlusion = sm.imresize(occlusion,(scale, scale))
        occ_y = y_val + 40
        occ_x = x_val + 30
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

    ranges = np.array([[0,300,900,680],[0,310,900,680], [0,321,900,680],[0,370,900,680],[0,0,900,170],[0,0,900,160],[0,0,900,230]])

    coord_offset = np.array([0,0,-200*1.5,-200*1.5])

    ranges[:] = ranges[:] + coord_offset

    return poke_info, scene_info, ranges

def generateImage(numb_pokemon, allowOcclusion, allowScale):
    random.seed()
    if numb_pokemon > 5:
        numb_pokemon = 5    
    poke_info, scene_info, ranges = load_data()      

    poke_def_size = 200
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
    scene_name = ''.join(['scenes/',str(scene_file),'.png'])

    #Setting up the background 

    background = cv2.imread(scene_name)
    background = sm.imresize(background, (680,1080))
    
    print seed_index

    poke_str = ''.join(['PokeSprites/',str(seed_index),'.png'])
    p = cv2.imread(poke_str, -1)
    p =  sm.imresize(p,(poke_def_size,poke_def_size))
    if allowScale == 1:
        #scale = poke_info[seed_index][1]
        p = sm.imresize(p,scale)
    x_val = random.randint(ranges[scene_val][0],ranges[scene-1][2])
    y_val = random.randint(ranges[scene_val][1],ranges[scene-1][3])

    background = overlayImage(background,p,y_val,x_val,p.shape[0])
    
    background = generate_occlusion(allowOcclusion, background,scene,p.shape[0]/2, y_val, x_val)
    
    #print scene_info[scene]
    for p_index in range(1,numb_pokemon):
        random.seed(p_index*10000)
        p_generated = random.choice(scene_info[scene])
        poke_str = ''.join(['PokeSprites/',str(p_generated),'.png'])
        p = cv2.imread(poke_str, -1)
        p = sm.imresize(p,(poke_def_size,poke_def_size))
        print p_generated;
        if allowScale == 1:
            scale = poke_info[seed_index][1]
            scale = 1.5
            p = sm.imresize(p,scale)
        x_val = random.randint(ranges[scene_val][0],ranges[scene-1][2])
        y_val = random.randint(ranges[scene_val][1],ranges[scene-1][3])
        background = overlayImage(background,p,y_val,x_val,p.shape[0])
        background = generate_occlusion(allowOcclusion, background,scene,p.shape[0]/2, y_val, x_val)
        

    return background

# Forest 1
# Ocean 2
# Mountain 3
# Grassland 4

im = generateImage(2,0,0)
cv2.imwrite('out2.png', im)
