import os
import time

import numpy as np
from multiprocessing import Pool
import librosa
import scipy.signal
import sklearn
import cv2
import matplotlib.pyplot as plt
datasets_dir = "input2/input"
output_root_dir = "inputim3"
divs = 64
sr = 32000
WIDTH = 313
n_mels = 128
frame_period = 5.0
n_frames = 128
fmin = 20
fmax = sr//2
imgs = 0

def rms(x):
    return np.sqrt(np.sum(x**2/len(x)))

def preEmphasis(signal, p=0.97):
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, signal)

def pitch_shift(x, cents=0):
    return librosa.effects.pitch_shift(x, sr, n_steps=1/1200*cents)

def time_stretch(x, r=1.):
    return librosa.effects.time_stretch(x, r)

def addNoise(x):
    return x + np.random.normal(0, 0.001, x.shape)

def rescale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))*255.

def process(folder):
    global imgs
    export_dir = folder
    os.makedirs(os.path.join(output_root_dir,export_dir), exist_ok=True)

    for file in glob.glob(folder + '/*.wav'):
        X, _ = librosa.load(file, sr=sr, mono=True)
        X *= 1. / max(0.01, np.max(np.abs(X)))

        for i in range(1):
            #print(file,i)
            ts_rate = 1 + (np.random.rand() - 0.5)*0.05
            ps_rate = (np.random.rand() - 0.5)*400
            if i==0:
                ts_rate = 1
                ps_rate = 0
            #x = time_stretch(X, ts_rate)
            #x = pitch_shift(x, cents=ps_rate)
            x = preEmphasis(X)
            #x = addNoise(x)
            x = x / np.max(np.abs(x))
            print(file)
            S = librosa.feature.melspectrogram(x, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)
            log_S = librosa.power_to_db(S)

            log_S = rescale(log_S)

            img = np.stack([log_S, log_S, log_S],axis=-1)

            img = img.reshape(n_mels, log_S.shape[1], 3)
            for j  in range(3):
                img[:,:,j] = img[:,:,j] / np.max(np.abs(img[:,:,j]))
            img *= 255
            start = 0
            while start<img.shape[1]:
                tmp = 0
                if start+WIDTH>img.shape[1]:
                    tmp = img[:,img.shape[1]-WIDTH:img.shape[1]]
                    start+=WIDTH
                else:
                    tmp = img[:,start:start+WIDTH]
                    start+=WIDTH
                bird = os.path.basename(folder)
                fileName = os.path.basename(file)[:-4]
                cv2.imwrite(os.path.join(output_root_dir,export_dir,"{}-{}-{:02d}-{:04d}.jpg".format(bird, fileName, i ,start)),tmp)


if __name__ == '__main__':

    folders = glob.glob(datasets_dir+"/*")
    TIME= time.time()
    cores = min(len(folders), 8)
    parallel = True
    print(folders)
    if parallel:
        p = Pool(cores)
        p.map(process, folders)

        p.close()
    else:
        for f in folders:
            process(f)

    print(time.time()-TIME)
