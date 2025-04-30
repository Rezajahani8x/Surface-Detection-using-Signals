import numpy as np
from scipy import stats as st
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def loadTrial_Train(dataFolder,id):
    xt = np.genfromtxt('{}trial{:02d}.x.t.csv'.format(dataFolder,id),delimiter=',')
    xv = np.genfromtxt('{}trial{:02d}.x.v.csv'.format(dataFolder,id),delimiter=',')
    yt = np.genfromtxt('{}trial{:02d}.y.t.csv'.format(dataFolder,id),delimiter=',')
    yv = np.genfromtxt('{}trial{:02d}.y.v.csv'.format(dataFolder,id),delimiter=',')
    yv = yv.astype(int)

    return xt, xv, yt, yv

def extractWin(xt,xv,yt,yv,winSz,timeStep):
   
    timeStart = np.max((np.min(xt),np.min(yt)))
    timeEnd = np.min((np.max(xt),np.max(yt)))

    t0 = timeStart
    t1 = t0 + winSz

    X = []
    Y = []
    while(t1 <= timeEnd):
      
        indices_x = (xt>=t0) * (xt<=t1)
        xWin = xv[indices_x]

        indices_y = (yt>=t0) * (yt<=t1)
        yWin = yv[indices_y]
        Y.append(st.mode(yWin).mode)

        X.append(xWin)

        t0 = t0 + timeStep
        t1 = t0 + winSz

    return np.array(X), np.array(Y)
    

def load_data(dataFolder,winSz,timeStep,idlist):

    for k,id in enumerate(idlist):
        xt, xv, yt, yv = loadTrial_Train(dataFolder,id=id)

        X, Y = extractWin(xt,xv,yt,yv,winSz,timeStep)

        if(k==0):
            X_data = X
            Y_data = Y
        else:
            X_data = np.concatenate((X_data,X),axis=0)
            Y_data = np.concatenate((Y_data,Y),axis=0)

    return X_data, Y_data

def extract_features(xt,xv,winSz,timeStep,timeStart,timeEnd):

    timeStart = np.min(xt)
    timeEnd = np.max(xt)

    t0 = timeStart
    t1 = t0 + winSz

    X = []
    
    while(t1 <= timeEnd):
      
        indices_x = (xt>=t0) * (xt<=t1)
        xWin = xv[indices_x]

        X.append(xWin)

        t0 = t0 + timeStep
        t1 = t0 + winSz
        

    return np.array(X)

def summaryPerf(yTrain,yTrainHat,y,yHat):
    
    cm = metrics.confusion_matrix(y,yHat,normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=
                                  ['Walk Hard','Down Stairs','Up Stairs','Walk Soft'])
    disp.plot()

    print('Training:  Acc = {:4.3f}'.format(metrics.accuracy_score(yTrain,yTrainHat)))
    print('Training:  BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(yTrain,yTrainHat)))
    print('Validation: Acc = {:4.3f}'.format(metrics.accuracy_score(y,yHat)))
    print('Validation: BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(y,yHat)))

