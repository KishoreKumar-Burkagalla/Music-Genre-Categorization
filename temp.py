import numpy as np
# import scipy.io.wavfile as wav
from librosa import load
from librosa.feature import mfcc

directory="./data/genres/"
folder="blues"
file="blues.00000.wav"
sig, rate = load(directory+folder+"/"+file)
print(rate)
#wav.read(directory+folder+"/"+file)
# computing MFCC features using librosa
mfcc_feat = mfcc(y=sig, sr=rate, n_fft=512)
print(mfcc_feat.shape)
# Mean and Covariance matrix to assist with distance computation
covariance = np.cov(mfcc_feat)
mean_matrix = mfcc_feat.mean(axis=1, dtype=np.float64)

feature = (mean_matrix, covariance)
print(mean_matrix.shape, covariance.shape, sep='\n')