#sys843 -1

import os
import fnmatch
import numpy as np
import argparse
import sys

from sys     import argv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from load_features     import load_all
from calc_scores       import calc_scores
from write_predictions import write_predictions
from scaling           import MinMaxScaling
from scaling           import GetStdRatio
from scaling           import ApplyScaling
from scaling           import DecimalScaling
from scaling           import MeanNormalization
from filter            import MedianFilter
from postprocessing    import OptimizePostProcessing

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras import optimizers
from keras.layers import Dropout
import tensorflow as tf

from sklearn.model_selection import KFold
from scaling import MeanNormalization
from load_features     import load_all


conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
import keras.backend.tensorflow_backend as tf_bkend
tf_bkend.set_session(sess)

delay = 2.8
sr_labels = 0.04
shift = int(np.round(delay/sr_labels))
shift = np.ones(1,dtype=int)*shift
path_features   = "/data/databases/Recola2016/features_audio/arousal/"


files_train = fnmatch.filter(os.listdir(path_features), "train*.csv")  # Filenames are the same for audio, video, text & labels
files_devel = fnmatch.filter(os.listdir(path_features), "dev*.csv")

print "Loading features..."
trainData   = load_all( files_train, [path_features], shift )
develData   = load_all( files_devel, [path_features], shift )

trainLabelsA1 = load_all( files_train, [ "/data/databases/Recola2016/labels/A1/" ] )
trainLabelsA2 = load_all( files_train, [ "/data/databases/Recola2016/labels/A2/" ] )
trainLabelsA3 = load_all( files_train, [ "/data/databases/Recola2016/labels/A3/" ] )
trainLabelsA4 = load_all( files_train, [ "/data/databases/Recola2016/labels/A4/" ] )
trainLabelsA5 = load_all( files_train, [ "/data/databases/Recola2016/labels/A5/" ] )
trainLabelsA6 = load_all( files_train, [ "/data/databases/Recola2016/labels/A6/" ] )
trainLabelsGS = load_all( files_train, [ "/data/databases/Recola2016/labels/gold_standard/" ] )

develLabelsA1 = load_all( files_devel, [ "/data/databases/Recola2016/labels/A1/" ] )
develLabelsA2 = load_all( files_devel, [ "/data/databases/Recola2016/labels/A2/" ] )
develLabelsA3 = load_all( files_devel, [ "/data/databases/Recola2016/labels/A3/" ] )
develLabelsA4 = load_all( files_devel, [ "/data/databases/Recola2016/labels/A4/" ] )
develLabelsA5 = load_all( files_devel, [ "/data/databases/Recola2016/labels/A5/" ] )
develLabelsA6 = load_all( files_devel, [ "/data/databases/Recola2016/labels/A6/" ] )
develLabelsGS = load_all( files_devel, [ "/data/databases/Recola2016/labels/gold_standard/" ] )


print "Normalize data (0-1 normalization)..."
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(trainData)
trainData = minMaxScaler.transform(trainData)
develData = minMaxScaler.transform(develData)

#Compute the prediction for each annotator
def Train(complexity,trainData,trainLabels):
    seed = 1
    reg = svm.LinearSVR(C=complexity,epsilon=0.1,random_state=seed,loss='squared_epsilon_insensitive')
    reg.fit(trainData,trainLabels)
    return reg

def Normalize(pred,devData,devLabels):
     # std prediction normalization
    #predTrain = reg.predict(trainData)
    ratio     = GetStdRatio(devLabels[:,0],pred)
    pred      = ApplyScaling(pred,ratio)

    # Mean bias normalization
    pred = MeanNormalization(devLabels[:,0],pred)

    return pred

def Normalize(pred,devData,devLabels):
     # std prediction normalization
    #predTrain = reg.predict(trainData)
    ratio     = GetStdRatio(devLabels[:,0],pred)
    pred      = ApplyScaling(pred,ratio)

    # Mean bias normalization
    pred = MeanNormalization(devLabels[:,0],pred)

    return pred
	
def NormalizeF(pred,Data,Labels):
     # std prediction normalization
    #predTrain = reg.predict(trainData)
    ratio     = GetStdRatio(Labels,pred)
    pred      = ApplyScaling(pred,ratio)

    # Mean bias normalization
    pred = MeanNormalization(Labels,pred)

    return pred


def PredictAndNormalize(reg,devData,devLabels):
    pred = reg.predict(devData)
    return Normalize(pred,devData,devLabels)

#SVR on DEv
regA1  = Train(0.00048828125,trainData,trainLabelsA1[:,0])
predA1 = PredictAndNormalize(regA1,develData,develLabelsA1)

scores_devel_A1 =  calc_scores(develLabelsA1[:,0],predA1)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA1)
print "A1: score :{}; score on gold standard: {}".format(scores_devel_A1[0],scores_devel_GS[0])




regA2  = Train(0.000244140625,trainData,trainLabelsA2[:,0])
predA2 = PredictAndNormalize(regA2,develData,develLabelsA2)

scores_devel_A2 =  calc_scores(develLabelsA2[:,0],predA2)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA2)
print "A2: score :{}; score on gold standard: {}".format(scores_devel_A2[0],scores_devel_GS[0])

regA2  = Train(0.000177779042372 ,trainData,trainLabelsA2[:,0])
predA2 = PredictAndNormalize(regA2,develData,develLabelsA2)

scores_devel_A2 =  calc_scores(develLabelsA2[:,0],predA2)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA2)
print "A2: score :{}; score on gold standard: {}".format(scores_devel_A2[0],scores_devel_GS[0])

sys.exit()
regA3  = Train(0.001953125,trainData,trainLabelsA3[:,0])
predA3 = PredictAndNormalize(regA3,develData,develLabelsA3)

scores_devel_A3 =  calc_scores(develLabelsA3[:,0],predA3)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA3)
print "A3: score :{}; score on gold standard: {}".format(scores_devel_A3[0],scores_devel_GS[0])

regA4  = Train(3.0517578125e-05,trainData,trainLabelsA4[:,0])
predA4 = PredictAndNormalize(regA4,develData,develLabelsA4)

scores_devel_A4 =  calc_scores(develLabelsA4[:,0],predA4)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA4)
print "A4: score :{}; score on gold standard: {}".format(scores_devel_A4[0],scores_devel_GS[0])

regA5  = Train(6.103515625e-05,trainData,trainLabelsA5[:,0])
predA5 = PredictAndNormalize(regA5,develData,develLabelsA5)

scores_devel_A5 =  calc_scores(develLabelsA5[:,0],predA5)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA5)
print "A5: score :{}; score on gold standard: {}".format(scores_devel_A5[0],scores_devel_GS[0])

regA6  = Train(0.000244140625,trainData,trainLabelsA6[:,0])
predA6 = PredictAndNormalize(regA6,develData,develLabelsA6)

scores_devel_A6 =  calc_scores(develLabelsA6[:,0],predA6)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA6)
print "A6: score :{}; score on gold standard: {}".format(scores_devel_A6[0],scores_devel_GS[0])

regGS  = Train(0.00048828125,trainData,trainLabelsGS[:,0])
predGS = PredictAndNormalize(regGS,develData,develLabelsGS)

scores_devel_A1 =  calc_scores(develLabelsA1[:,0],predGS)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predGS)
print "GS: score :{}; score on gold standard: {}".format(scores_devel_GS[0],scores_devel_GS[0])


print "new values+++++++++++++++++++++++++++++++++++"
regA1  = Train(0.00462894122312,trainData,trainLabelsA1[:,0])
predA1 = PredictAndNormalize(regA1,develData,develLabelsA1)
#predA1 = MedianFilter(predA1,63)
scores_devel_A1 =  calc_scores(develLabelsA1[:,0],predA1)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA1)
print "A1: score :{}; score on gold standard: {}".format(scores_devel_A1[0],scores_devel_GS[0])


regA2  = Train(0.000638845682982,trainData,trainLabelsA2[:,0])
predA2 = PredictAndNormalize(regA2,develData,develLabelsA2)
#predA2 = MedianFilter(predA2,115)
scores_devel_A2 =  calc_scores(develLabelsA2[:,0],predA2)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA2)
print "A2: score :{}; score on gold standard: {}".format(scores_devel_A2[0],scores_devel_GS[0])

regA3  = Train(0.000847745221316,trainData,trainLabelsA3[:,0])
predA3 = PredictAndNormalize(regA3,develData,develLabelsA3)
#predA3 = MedianFilter(predA3,99)
scores_devel_A3 =  calc_scores(develLabelsA3[:,0],predA3)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA3)
print "A3: score :{}; score on gold standard: {}".format(scores_devel_A3[0],scores_devel_GS[0])


regA4  = Train(0.00198094997455,trainData,trainLabelsA4[:,0])
predA4 = PredictAndNormalize(regA4,develData,develLabelsA4)
#predA4 = MedianFilter(predA4,59)
scores_devel_A4 =  calc_scores(develLabelsA4[:,0],predA4)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA4)
print "A4: score :{}; score on gold standard: {}".format(scores_devel_A4[0],scores_devel_GS[0])


regA5  = Train(0.000362791574496,trainData,trainLabelsA5[:,0])
predA5 = PredictAndNormalize(regA5,develData,develLabelsA5)
#predA5 = MedianFilter(predA5,267)
scores_devel_A5 =  calc_scores(develLabelsA5[:,0],predA5)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA5)
print "A5: score :{}; score on gold standard: {}".format(scores_devel_A5[0],scores_devel_GS[0])


regA6  = Train(0.000155256229367,trainData,trainLabelsA6[:,0])
predA6 = PredictAndNormalize(regA6,develData,develLabelsA6)
#predA6 = MedianFilter(predA6,151)
scores_devel_A6 =  calc_scores(develLabelsA6[:,0],predA6)
scores_devel_GS =  calc_scores(develLabelsGS[:,0],predA6)
print "A6: score :{}; score on gold standard: {}".format(scores_devel_A6[0],scores_devel_GS[0])


sys.exit()

SVRPredictions = np.stack((predA1,predA2,predA3,predA4,predA5,predA6),axis=1)

#predict TrainData on SVR
tpredA1 = PredictAndNormalize(regA1,trainData,trainLabelsA1)



tpredA2 = PredictAndNormalize(regA2,trainData,trainLabelsA2)

tpredA3 = PredictAndNormalize(regA3,trainData,trainLabelsA3)

tpredA4 = PredictAndNormalize(regA4,trainData,trainLabelsA4)

tpredA5 = PredictAndNormalize(regA5,trainData,trainLabelsA5)

tpredA6 = PredictAndNormalize(regA6,trainData,trainLabelsA6)

#LSTM for TrainData

print "Now we predict training data itself with the LSTM we have built to produce predictions for later learning"
'''
def create_seq(l, win): #win=30
	l = list(l)

	lpadded = win // 2 * [l[0]] + l + win // 2 * [l[-1]]
	out = [lpadded[i:(i + win)] for i in range(len(l))]

	assert len(out) == len(l)
	return out

seq = []

chunk = int(len(trainLabelsGS)/9)
for s in range(0,9):
	seq += create_seq(range(s*chunk, (s+1)*chunk), 30)
	
train_seq = trainData[seq]
dev_seq= develData[seq]

model = Sequential()               
model.add(LSTM(100, return_sequences=False, input_shape=(train_seq.shape[1],train_seq.shape[2]))) # the first layer of model which has 100 layer of 
#model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='mse')

#A1			  
model.fit(train_seq, trainLabelsA1[:,0], epochs=30, batch_size=256)

tpred_A1 = model.predict(train_seq)

np.save('tpred_A1',tpred_A1)

#A2

model.compile(optimizer=sgd, loss='mse')

model.fit(train_seq, trainLabelsA2[:,0], epochs=30, batch_size=256)

tpred_A2 = model.predict(train_seq)

np.save('tpred_A2',tpred_A2)
#A3
model.compile(optimizer=sgd, loss='mse')

model.fit(train_seq, trainLabelsA3[:,0], epochs=30, batch_size=256)

tpred_A3 = model.predict(train_seq)

np.save('tpred_A3',tpred_A3)

#A4
model.compile(optimizer=sgd, loss='mse')

model.fit(train_seq, trainLabelsA4[:,0], epochs=30, batch_size=256)

tpred_A4 = model.predict(train_seq)

np.save('tpred_A4',tpred_A4)

#A5

model.compile(optimizer=sgd, loss='mse')
model.fit(train_seq, trainLabelsA5[:,0], epochs=30, batch_size=256)

tpred_A5 = model.predict(train_seq)

np.save('tpred_A5',tpred_A5)

#A6

model.compile(optimizer=sgd, loss='mse')
model.fit(train_seq, trainLabelsA6[:,0], epochs=30, batch_size=256)

tpred_A6 = model.predict(train_seq)

np.save('tpred_A6',tpred_A6)
'''

tpred_A1 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/tpred_A1.npy')
tpred_A2 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/tpred_A2.npy')
tpred_A3 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/tpred_A3.npy')
tpred_A4 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/tpred_A4.npy')
tpred_A5 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/tpred_A5.npy')
tpred_A6 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/tpred_A6.npy')




trainpred1=np.stack((tpredA1,tpredA2,tpredA3,tpredA4,tpredA5,tpredA6),axis=1)
trainpred=np.concatenate((trainpred1,tpred_A1,tpred_A2,tpred_A3,tpred_A4,tpred_A5,tpred_A6),axis=1)


#loading predictions of Develdata and normalize it with mean normalization
pred_A1 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/pred_A1.npy')
pred_A1 = MeanNormalization(develLabelsGS[:,0],pred_A1)

pred_A2 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/pred_A2.npy')
pred_A2 = MeanNormalization(develLabelsGS[:,0],pred_A2)

pred_A3 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/pred_A3.npy')
pred_A3 = MeanNormalization(develLabelsGS[:,0],pred_A3)

pred_A4 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/pred_A4.npy')
pred_A4 = MeanNormalization(develLabelsGS[:,0],pred_A4)

pred_A5 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/pred_A5.npy')
pred_A5 = MeanNormalization(develLabelsGS[:,0],pred_A5)

pred_A6 = np.load('/home/AN79510/Recola/Winter_2018/Rafi/pred_A6.npy')
pred_A6 = MeanNormalization(develLabelsGS[:,0],pred_A6)


#creating a matrix of all predictions from SVR and LSTM



devpred=np.concatenate((SVRPredictions,pred_A1,pred_A2,pred_A3,pred_A4,pred_A5,pred_A6), axis=1)


#printing the separate predictions by LSTM:
print "now we print the score we achieved through each model based on each annotator with LSTM:"

score_A1 =  calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(pred_A1[:,0],develData,develLabelsGS),131))
print "Using LSTM , label A1 the CCC is:"
print score_A1

score_A2 =  calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(pred_A2[:,0],develData,develLabelsGS),131))
print "Using LSTM , label A2 the CCC is:"
print score_A2

score_A3 =  calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(pred_A3[:,0],develData,develLabelsGS),131))
print "Using LSTM , label A3 the CCC is:"
print score_A3

score_A4 =  calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(pred_A4[:,0],develData,develLabelsGS),131))
print "Using LSTM , label A4 the CCC is:"
print score_A4

score_A5=  calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(pred_A5[:,0],develData,develLabelsGS),131))
print "Using LSTM , label A5 the CCC is:"
print score_A5

score_A6=  calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(pred_A6[:,0],develData,develLabelsGS),131))
print "Using LSTM , label A6 the CCC is:"
print score_A6

#Fusion:::

print "creating a learning model for aggregation section of our ensemble model:"

#finding the optimum parameter C for our learning model
#Rafi fusion

num_steps = 16
complexities = np.logspace(-15,0,num_steps,base=2.0)
scores_train = np.empty((32,3))
seed =0
kf = KFold(n_splits=2)
arousalLabels = trainLabelsGS[:,0]
i=0
Ci=np.empty((32,))

for train_index,test_index in kf.split(develData):
        trainFusion,testFusion = trainpred[train_index],trainpred[test_index]
        trainFusionLabels,testFusionLabels = arousalLabels[train_index],arousalLabels[test_index]
	for comp in range(len(complexities)):
		regA = svm.LinearSVR(C=complexities[comp],epsilon=0.1,random_state=seed)
		regA.fit(trainFusion,trainFusionLabels)
		predA = regA.predict(testFusion)
		scores_train[i,:]= calc_scores(testFusionLabels,MedianFilter(NormalizeF(predA,testFusion,testFusionLabels),131))
		Ci[i]=comp
		i=i+1




ind= np.argmax(scores_train[:,0])
C_o=Ci[ind]

print "optimum value for parameter C for last Fusion:"
print C_o

regF = svm.LinearSVR(C=C_o,epsilon=0.1,random_state=seed)
regF.fit(trainpred,trainLabelsGS[:,0])
predFusion = PredictAndNormalize(regF,devpred,develLabelsGS)

score_fusio= calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(predFusion,develData,develLabelsGS),131))

print "the score for training on the prediction on traindata and testing on the prediction of develdata with both SVR and LASTM:"
print score_fusio


# Average fusion


ave_pred=np.empty((len(devpred),1))
c=0
for frame in devpred:
    ave_pred[c][0]=(np.sum(frame))/12
    c=c+1

score_ave=  calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(ave_pred[:,0],develData,develLabelsGS),131))
print "using simple average on predictions made with SVR and LSTM, the score will be:"	
print score_ave

#Patrick Fusion

num_steps = 16
complexities = np.logspace(-15,0,num_steps,base=2.0)
scores_train = np.empty((num_steps,3))
seed =0
predictions = np.zeros(len(develData))
kf = KFold(n_splits=9)
arousalLabels = develLabelsGS[:,0]

for comp in range(2,3):
    speakerAccuracies = [0]*9
    spk=0
    for train_index,test_index in kf.split(develData):
        trainFusion,testFusion = devpred[train_index],devpred[test_index]
        trainFusionLabels,testFusionLabels = arousalLabels[train_index],arousalLabels[test_index]

        reg = svm.LinearSVR(C=complexities[comp],epsilon=0.1,random_state=seed)
        reg.fit(trainFusion,trainFusionLabels)
        pred = reg.predict(testFusion)

        predictions[test_index] = pred

    score = calc_scores(develLabelsGS[:,0],MedianFilter(Normalize(predictions,develData,develLabelsGS),131))
    print "Fusion as patrick CCC : {} (complexities: {})".format(score[0],complexities[comp])







