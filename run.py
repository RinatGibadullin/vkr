import os
import sys

import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.ndimage import median_filter
import sklearn.cluster
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

from SALAMI import get_labels
from serato_markers import create_serato_markers

from os import listdir
from os.path import isfile, join

Decoder = {
    0: "Не определен",
    1: "Вступление",
    2: "Куплет",
    3: "Предприпев",
    4: "Припев",
    5: "Соло",
    6: "Окончание"
}

def get_times_and_labels(audio_name):
    audio_data = audio_name
    x, sr = librosa.load(audio_data)

    mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=512, n_mfcc=14)

    audio_harmonic, _ = librosa.effects.hpss(x)
    pcp_cqt = np.abs(librosa.hybrid_cqt(audio_harmonic)) ** 2
    pcp = librosa.feature.chroma_cqt(C=pcp_cqt)

    # plt.imshow(pcp[:, 0:2029], aspect="auto")
    # plt.show()

    tempo, beats = librosa.beat.beat_track(y=x, sr=sr, trim=False)
    beats = librosa.util.fix_frames(beats, x_max=pcp.shape[1])
    pcp = librosa.util.sync(pcp, beats,
                            aggregate=np.median)

    # plt.imshow(pcp[:, 0:100], aspect="auto")
    # plt.show()
    beats = librosa.util.fix_frames(beats, x_max=mfcc.shape[1])
    mfcc = librosa.util.sync(mfcc, beats)

    # plt.imshow(mfcc, aspect="auto")
    # plt.show()

    ##########################################################
    # преобразование PCP
    ###########################################################

    R = librosa.segment.recurrence_matrix(pcp,
                                          width=9,
                                          mode='affinity',
                                          metric='cosine',
                                          sym=True)

    # plt.imshow(R, aspect="auto")
    # plt.show()

    df = librosa.segment.timelag_filter(median_filter)
    Rf = df(R, size=(1, 9))
    # plt.imshow(Rf, aspect="auto")
    # plt.show()

    ##########################################################
    # преобразование MFCC
    ###########################################################
    path_distance = np.sum(np.diff(mfcc, axis=1) ** 2, axis=0)

    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    # plt.plot(path_sim)
    # plt.title("path_sim")
    # plt.show()

    # MFCC массив в матрицу с данными около диагонали
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    ##########################################################
    # Объединение
    ##########################################################

    # MFCC матрицу суммируем по x, тем самым получаем массив,
    # где каждый элемент (такт) MFCC массива это сумма с предыдущим элементом (тактом)
    deg_path = np.sum(R_path, axis=1)
    # plt.plot(deg_path)
    # plt.title("deg_path")
    # plt.show()

    # PCP матрицу суммируем по x
    deg_rec = np.sum(Rf, axis=1)
    # plt.plot(deg_rec)
    # plt.title("deg_rec PCP")
    # plt.show()

    PCPandMFCC = deg_path + deg_rec
    mu = deg_path.dot(PCPandMFCC) / np.sum((PCPandMFCC) ** 2)

    PCPMatrixWithMU = mu * Rf
    # plt.imshow(PCPMatrixWithMU)
    # plt.title("PCPMatrixWithMU")
    # plt.show()

    MFCCMatrixWithMU = (1 - mu) * R_path
    # plt.imshow(MFCCMatrixWithMU)
    # plt.title("MFCCMatrixWithMU")
    # plt.show()

    A = PCPMatrixWithMU + MFCCMatrixWithMU

    plt.imshow(A, aspect="auto")
    plt.show()

    #####################################################
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    plt.imshow(L, aspect="auto")
    plt.title("L")
    plt.show()

    evals, evecs = scipy.linalg.eigh(L)

    plt.imshow(evecs[:, :50], aspect="auto")
    plt.title("evecs")
    plt.show()

    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    plt.imshow(evecs[:, :50], aspect="auto")
    plt.title("evecs median")
    plt.show()

    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5
    plt.imshow(Cnorm, aspect="auto")
    plt.title("Cnorm")
    plt.show()

    k = 4

    evecs_n = evecs[:, :10]
    Cnorm_n = Cnorm[:, 9:10] + 1e-5

    plt.imshow(evecs_n, aspect="auto")
    plt.title("evecs_n")
    plt.show()

    plt.imshow(Cnorm_n, aspect="auto")
    plt.title("Cnorm_n")
    plt.show()

    x_data = evecs_n / Cnorm_n
    plt.imshow(x_data, aspect="auto")
    plt.title("X")
    plt.show()

    # сохранение инфы о треке
    frame_times = librosa.frames_to_time(beats, sr=sr)[:-1]
    # # tick_hms = np.array([str(time.strftime("%M.%S", time.gmtime(s))) for s in frame_times])
    # beat_times = np.array([float('{:.1f}'.format(s)) for s in frame_times])
    # # beat_times = np.round(frame_times, 2)
    # beat_times = np.array([beat_times]).T.astype(str)
    # conc = np.concatenate((x_data, beat_times), axis=1)
    # np.savetxt("12.csv", conc, delimiter=",", fmt="%s")

    # KM = sklearn.cluster.KMeans(n_clusters=k, n_init=50, max_iter=500)
    # если взять просто evecs_n вместо x_data то лучше и k взять 10
    # seg_ids = KM.fit_predict(x_data)

    # df = pd.read_csv('data/data.csv', delimiter=',', header=None)
    df = pd.read_csv('data/12.csv', delimiter=',', header=None)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, Y_train)
    print('Accuracy: {:.2f}'.format(classifier.score(X_test, Y_test)))

    seg_ids = classifier.predict(x_data)

    seg = np.copy(seg_ids)
    plt.plot(seg)
    plt.show()

    count = 0
    dif_index = 0
    for idx, item in enumerate(seg_ids):
        if len(seg_ids) != idx:
            if seg_ids[idx] != seg_ids[idx - 1]:
                if count < 8:
                    seg_ids[dif_index: idx] = seg_ids[dif_index - 1]
                count = 0
                dif_index = idx
            else:
                count += 1
    plt.plot(seg_ids)
    plt.show()

    id1 = seg_ids[:-1]
    id2 = seg_ids[1:]
    ids = seg_ids[:-1] != seg_ids[1:]
    npy = np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beats 0 as a boundary
    bound_idxs = bound_beats
    bound_idxs = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    bound_segs = list(seg_ids[bound_idxs])

    # Tack on the end-time
    # bound_idxs = list(np.append(bound_idxs, len(Cnorm) - 1))

    frame_times = librosa.frames_to_time(beats, sr=sr)
    est_times = frame_times[bound_idxs]
    est_labels = bound_segs

    return est_times, est_labels

if __name__ == '__main__':
    directory_path = "audio/"
    audio_paths = [join(directory_path, f) for f in listdir(directory_path) if os.path.splitext(f)[-1].lower() == '.mp3']

    for audio_path in audio_paths:
        print("analyzing " + audio_path)
        times, labels = get_times_and_labels(audio_path)

        result = list()
        for idx, time in enumerate(times):
            row = [time, Decoder[labels[idx]]]
            result.append(row)
        res = np.asarray(result)

        audiofile_name = os.path.splitext(audio_path)[0].split('/')[-1]
        np.savetxt(directory_path + audiofile_name + ".txt", res, delimiter="\t", fmt="%s")

        # second to milliseconds
        times = (times * 1000).astype(int)
        create_serato_markers(times, labels, audio_path)

    sys.exit()