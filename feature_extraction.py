import os
import pickle
import numpy as np
# import scipy.io.wavfile as wav
from librosa import load
from librosa.feature import mfcc

def preprocess(directory="C:/Project/", filename="gtzan.dat"):
    f = open(filename ,'wb')
    i=0
    for folder in os.listdir(directory):
        i += 1
        if i == 11 :
            break 	
        for file in os.listdir(directory+folder):
            print("Working with: " + directory + folder + "/" + file)
            # Reading the audio file
            sig, rate = load(directory+folder+"/"+file, sr=None)
            #wav.read(directory+folder+"/"+file)
            # computing MFCC features using librosa
            mfcc_feat = mfcc(y=sig, sr=rate)
            # Mean and Covariance matrix to assist with distance computation
            covariance = np.cov(mfcc_feat)
            mean_matrix = mfcc_feat.mean(axis=1, dtype=np.float64)
            
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
    f.close()

def loadDataset(filename="gtzan.dat"):
    if not os.path.exists(filename):
        preprocess(filename=filename)
    data, labels = [], []
    with open(filename,'rb') as f:
        while True:
            try:
                temp = pickle.load(f)
                data.append([temp[0]] + list(temp[1]))
                labels.append(temp[-1])
            except EOFError:
                f.close()
                break
    # Conversion to NumPy arrays for easier manipulation
    data, labels = np.array(data), np.array(labels)
    return data, labels
