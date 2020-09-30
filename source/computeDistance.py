from __future__ import print_function


import os
import fnmatch
import numpy as np
import argparse
import sys

from load_features     import load_all
from scipy.spatial     import distance

path_features   = "/home/pcardinal/Databases/Recola/features_audio/arousal/"

files_train = fnmatch.filter(os.listdir(path_features), "train*.csv")

#Train   = load_all( files_train, [path_features] )

dev = load_all([sys.argv[1]],[path_features])

#print dev
indexDev = 0
for devSample in dev:
    distances = []
    for filename in files_train:
        Train = load_all([filename],[path_features])
        train_id = os.path.splitext(filename)[0]
        index = 0
        for trainSample in Train:
            dist = distance.euclidean(devSample,trainSample)
            distances.append((train_id,index,dist))
            #print "{};{};{};{}".format(filename,index,indexDev,dist)
            index = index+1
        distances.sort(key=lambda tup: tup[2])
    print("{}".format(indexDev),end="")
    for d in distances[:20]:
        print(";{};{};{}".format(d[0],d[1],d[2]),end="")
    print("")
    #sys.exit(0)
    indexDev = indexDev+1
