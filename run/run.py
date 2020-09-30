

import os
import fnmatch
import numpy as np
import argparse

from sys     import argv
from sklearn import svm

from load_features     import load_all
from calc_scores       import calc_scores
from write_predictions import write_predictions
from scaling           import MinMaxScaling
from scaling           import GetStdRatio
from scaling           import ApplyScaling
from scaling           import DecimalScaling

from filter            import MedianFilter
import sys

# Set folders here
b_test_available      = False  # If the test labels are not available, the predictions on test are written into the folder 'path_test_predictions'

# Folders with provided features and labels
#path_audio_features = "audio_features_xbow_6s/"
path_audio_features   = "/data/databases/Recola/features_audio/"
path_ecg_features     = "/data/databases/Recola/features_ecg/"
path_eda_features     = "/data/databases/Recola/features_eda/"
path_video_geometric  = "/data/databases/Recola/features_video_geometric/"
path_video_appearance = "/data/databases/Recola/features_video_geometric/"
path_labels           = "/data/databases/Recola/ratings_gold_standard/"

sr_labels = 0.04

delay = 0.0
b_audio     = True
b_ecg       = False
b_eda       = False
b_video_geo = False
b_video_app = False

parser = argparse.ArgumentParser(description='run experiments on SEWA databases using SVR',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--audio',action='store_true',help="Use audio modality",default=False)
parser.add_argument('--ecg',action='store_true',help="Use ecg modality",default=False)
parser.add_argument('--eda',action='store_true',help="Use eda modality",default=False)
parser.add_argument('--video_geo',action='store_true',help="Use video_geometric modality",default=False)
parser.add_argument('--video_app',action='store_true',help="Use video_appearance modality",default=False)
parser.add_argument('--normalize',action='store_true',help="Normalize input",default=False)
parser.add_argument('--delay',type=float,help="Set delay compensation",default=0.0)
parser.add_argument('--saveDir',help="Directory where results will be save",default="./Results")
parser.add_argument('--saveDevPred',action='store_true',help="Save predictions, file by file, for the dev set",default=False)
parser.add_argument('--minMaxScaling',action='store_true',help="Apply min max normalization to predictions",default=False)
parser.add_argument('--stdScaling',action='store_true',help="Apply std scaling normalization to predictions",default=False)
parser.add_argument('--decimalScaling',action='store_true',help="Apply decimal scaling to predictions",default=False)
parser.add_argument('--epsilon',type=float,help="epsilon value for SVR",default=0.01)
parser.add_argument('--medianFilter',type=int,help="Apply median filtering to predictions",default=0)
args = parser.parse_args()

delay          = args.delay
b_audio        = args.audio
b_ecg          = args.ecg
b_eda          = args.eda
b_video_geo    = args.video_geo
b_video_app    = args.video_app
savedir        = args.saveDir
savedev        = args.saveDevPred
minMaxScaling  = args.minMaxScaling
stdScaling     = args.stdScaling
decScaling     = args.decimalScaling
epsilon        = args.epsilon
medianFilter   = args.medianFilter
normalize      = args.normalize
path_test_predictions = savedir+"/test_predictions/"
path_dev_predictions  = savedir+"/dev_predictions/"


modality_str = "Modality used: "
path_features = []
if b_audio:
    modality_str += "audio "
    path_features.append( path_audio_features )
if b_ecg:
    modality_str += "ECG "
    path_features.append( path_audio_features )
if b_eda:
    modality_str += "EDA "
    path_features.append( path_audio_features )
if b_video_geo:
    modality_str += "Video geometric "
    path_features.append( path_video_features )
if b_video_app:
    modality_str += "Video appearance "
    path_features.append( path_video_features )

print modality_str

#create results directory
if not os.path.exists(savedir):
    os.mkdir(savedir)
if not os.path.exists(path_test_predictions):
    os.mkdir(path_test_predictions)
if savedev and not os.path.exists(path_dev_predictions):
    os.mkdir(path_dev_predictions)

# Compensate the delay (quick solution)
shift = int(np.round(delay/sr_labels))
shift = np.ones(len(path_features),dtype=int)*shift

files_train = fnmatch.filter(os.listdir(path_features[0]), "train*.csv")  # Filenames are the same for audio, video, text & labels
files_devel = fnmatch.filter(os.listdir(path_features[0]), "dev*.csv")
files_test  = fnmatch.filter(os.listdir(path_features[0]), "test*.csv")

# Load features and labels
print "Loading features..."
Train   = load_all( files_train, path_features, shift )
Devel   = load_all( files_devel, path_features, shift )
Test    = load_all( files_test, path_features, shift, separate=True )  # Load test features separately to store the predictions in separate files

Train_L = load_all( files_train, [ path_labels ] )  # Labels are not shifted
Devel_L = load_all( files_devel, [ path_labels ] )

Test    = load_all( files_test, path_features, shift, separate=True )  # Load test features separately to store the predictions in separate files

print "Run experiments..."
# Run liblinear (scikit-learn)
# Optimize complexity
num_steps = 16
complexities = np.logspace(-15,0,num_steps,base=2.0)  # 2^-15, 2^-14, ... 2^0

scores_devel_A = np.empty((num_steps,3))
scores_devel_V = np.empty((num_steps,3))

seed = 0

for comp in range(0,num_steps):
    # Train and compute the score for arousal
    regA = svm.LinearSVR(C=complexities[comp],epsilon=epsilon,random_state=seed)
    regA.fit(Train,Train_L[:,0])
    predA = regA.predict(Devel)

    if minMaxScaling:
        predA = MinMaxScaling(Train_L[:,0],predA)
    if stdScaling:
        predTrain = regA.predict(Train)
        ratio     = GetStdRatio(Train_L[:,0],predTrain)
        predA     = ApplyScaling(predA,ratio)
    if decScaling:
        predA = DecimalScaling(Train_L[:,0],predA)
    if medianFilter > 0:
        predA = MedianFilter(predA,medianFilter)
    scores_devel_A[comp,:] = calc_scores(Devel_L[:,0],predA)

    # Train and compute de score for valence
    regV = svm.LinearSVR(C=complexities[comp],random_state=seed)
    regV.fit(Train,Train_L[:,1])
    predV = regV.predict(Devel)
    if minMaxScaling:
        predV = MinMaxScaling(Train_L[:,1],predV)
    if stdScaling:
        predTrain = regV.predict(Train)
        ratio     = GetStdRatio(Train_L[:,1],predTrain)
        predV     = ApplyScaling(predV,ratio)
    if decScaling:
        predV = DecimalScaling(Train_L[:,0],predV)
    if medianFilter > 0:
        predV = MedianFilter(predV,medianFilter)

    scores_devel_V[comp,:] = calc_scores(Devel_L[:,1],predV)

ind_opt_A = np.argmax(scores_devel_A[:,0])
ind_opt_V = np.argmax(scores_devel_V[:,0])
comp_opt_A = complexities[ind_opt_A]
comp_opt_V = complexities[ind_opt_V]

#if savedev:
#    dev   = load_all( files_devel, path_features, shift, separate=True )
#    for f in range(0,len(files_devel)):
#        predA = regA.predict(dev[f])
#        predV = regV.predict(dev[f])
#        predictions = np.array([predA,predV])
#        write_predictions(path_dev_predictions,files_devel[f],predictions,sr_labels)

# Print scores (CCC, PCC, RMSE) on the development set
print("Arousal devel (CCC,PCC,RMSE):")
print(scores_devel_A[ind_opt_A,:])
print("Valence devel (CCC,PCC,RMSE):")
print(scores_devel_V[ind_opt_V,:])

