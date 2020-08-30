import os
import time

import librosa
import numpy as np
import glob
import numpy as np
from multiprocessing import Pool

datasets_dir = "inpute"
output_root_dir = "input2"
divs = 16

def process(folder):
    sampling_rate = 32000
    num_mcep = 36
    frame_period = 5.0
    n_frames = 128
    X=[]
    Y=[]
    for file in glob.glob(folder + '/*.wav'):
        wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
        wav *= 1. / max(0.01, np.max(np.abs(wav)))

        wav_splitted = librosa.effects.split(wav,top_db=48)

        export_dir = folder
        os.makedirs(os.path.join(output_root_dir,export_dir), exist_ok=True)


        for s in range(wav_splitted.shape[0]):
            x = wav[wav_splitted[s][0]:wav_splitted[s][1]]
            X = np.concatenate([X,x],axis=0)
    try:
        X *= 1. / max(0.01, np.max(np.abs(X)))
        wavlen = X.shape[0]
        crop_size = wavlen // divs
        start = 0
        for i in range(divs):
            sub = 0
            if(i==divs-1):
                sub = X[start:]
            else:
                sub = X[start:start+crop_size]

            start += crop_size
            sub = sub.astype(np.float32)
            librosa.output.write_wav(os.path.join(output_root_dir,export_dir,"{}_".format(i)+os.path.basename(folder)+".wav"), sub, sampling_rate)
    except:
        print("{} has error".format(folder))


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
