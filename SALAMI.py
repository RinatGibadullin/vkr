import os
import sys

import librosa.display
import numpy as np
import scipy
from scipy.ndimage import median_filter
import pandas as pd
from os import listdir
from os.path import join

decoder = {
    "Intro": 1,
    "Verse": 2,
    "Pre-Chorus": 3,
    "Chorus": 4,
    "Solo": 5,
    "Outro": 6,
}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_labels(beat_times, annotion_file):
    df = pd.read_csv(annotion_file, delimiter=r"\s+", header=None)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    decodedY = list()
    for y in Y:
        if y in decoder:
            x = decoder[y]
        else:
            x = 0
        decodedY.append(x)

    labels = np.copy(beat_times)
    for idx, x in enumerate(np.asarray(X)):
        id = find_nearest(beat_times, float(x))
        labels[id : ] = decodedY[idx]
    labels = labels.astype(int)
    return labels

def save_data(audiofile_path):
    x, sr = librosa.load(audiofile_path)

    mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=512, n_mfcc=14)

    audio_harmonic, _ = librosa.effects.hpss(x)
    pcp_cqt = np.abs(librosa.hybrid_cqt(audio_harmonic)) ** 2
    pcp = librosa.feature.chroma_cqt(C=pcp_cqt)

    tempo, beats = librosa.beat.beat_track(y=x, sr=sr, trim=False)
    beats = librosa.util.fix_frames(beats, x_max=pcp.shape[1])
    pcp = librosa.util.sync(pcp, beats,
                            aggregate=np.median)

    beats = librosa.util.fix_frames(beats, x_max=mfcc.shape[1])
    mfcc = librosa.util.sync(mfcc, beats)

    ##########################################################
    # преобразование PCP
    ###########################################################

    R = librosa.segment.recurrence_matrix(pcp,
                                          width=9,
                                          mode='affinity',
                                          metric='cosine',
                                          sym=True)

    df = librosa.segment.timelag_filter(median_filter)
    Rf = df(R, size=(1, 9))

    ##########################################################
    # преобразование MFCC
    ###########################################################
    path_distance = np.sum(np.diff(mfcc, axis=1) ** 2, axis=0)

    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)

    # MFCC массив в матрицу с данными около диагонали
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    ##########################################################
    # Объединение
    ##########################################################

    # MFCC матрицу суммируем по x, тем самым получаем массив,
    # где каждый элемент (такт) MFCC массива это сумма с предыдущим элементом (тактом)
    deg_path = np.sum(R_path, axis=1)

    # PCP матрицу суммируем по x
    deg_rec = np.sum(Rf, axis=1)

    PCPandMFCC = deg_path + deg_rec
    mu = deg_path.dot(PCPandMFCC) / np.sum((PCPandMFCC) ** 2)

    PCPMatrixWithMU = mu * Rf
    MFCCMatrixWithMU = (1 - mu) * R_path
    A = PCPMatrixWithMU + MFCCMatrixWithMU

    #####################################################
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5

    evecs_n = evecs[:, :10]
    Cnorm_n = Cnorm[:, 9:10] + 1e-5
    x_data = evecs_n / Cnorm_n

    # сохранение инфы о треке
    audiofile_name = os.path.splitext(audiofile_path)[0].split('/')[-1]

    frame_times = librosa.frames_to_time(beats, sr=sr)[:-1]
    # labels = get_labels(frame_times, 'data/labels/' + audiofile_name + "/parsed" + "/textfile1_functions.txt")
    labels = get_labels(frame_times, 'my_audio_csv/12/parsed/textfile1_functions.txt')

    labels = np.array([labels]).T.astype(str)
    conc = np.concatenate((x_data, labels), axis=1)
    # np.savetxt('data/csv/' + audiofile_name + ".csv", conc, delimiter=",", fmt="%s")
    np.savetxt('data/' + audiofile_name + ".csv", conc, delimiter=",", fmt="%s")

if __name__ == '__main__':
    # directory_path = "data/audio/"
    # audio_paths = [join(directory_path, f) for f in listdir(directory_path) if os.path.splitext(f)[-1].lower() == '.mp3']

    # for audio_path in audio_paths:
    #     print("analyzing " + audio_path)
    #     save_data(audio_path)

    save_data("audio/12.mp3")

    # directory_path = 'data/csv/'
    # all_filenames = [join(directory_path, f) for f in listdir(directory_path) if os.path.splitext(f)[-1].lower() == '.csv']
    #
    # data = np.zeros(shape=(1, 11))
    #
    # for f in all_filenames:
    #     new_data = np.array(pd.read_csv(f, delimiter=','))
    #     data = np.concatenate((data, new_data), axis=0)
    #
    # np.savetxt("data/data.csv", data, delimiter=",", fmt="%s")
    sys.exit()