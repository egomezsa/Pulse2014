#!/usr/bin/python
import sys
import find_pokemon
import scipy.stats.mstats as mstats

res = find_pokemon.compare_image(sys.argv[1], int(sys.argv[2]))

print "Results are in"

count = dict()

for i in res:
    if i not in count:
        if i != 'r' or i != -1:
            count[i] = 1
    else:
        count[i] = count[i] + 1

print count
