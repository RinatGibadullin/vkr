import sys
import time

import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.ndimage import median_filter
import sklearn.cluster
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

from serato_markers import create_serato_markers

if __name__ == '__main__':
    audio_data = '8.mp3'
    x, sr = librosa.load(audio_data)

    mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=512, n_mfcc=14)


    audio_harmonic, _ = librosa.effects.hpss(x)
    pcp_cqt = np.abs(librosa.hybrid_cqt(audio_harmonic)) ** 2
    pcp = librosa.feature.chroma_cqt(C=pcp_cqt)


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
    path_distance = np.sum(np.diff(mfcc, prepend=mfcc[0][0], axis=1) ** 2, axis=0)

    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    # plt.plot(path_sim)
    # plt.title("path_sim")
    # plt.show()

    # MFCC массив в матрицу с данными около диагонали
    # R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    ##########################################################
    # Объединение
    ##########################################################

    # MFCC матрицу суммируем по x, тем самым получаем массив,
    # где каждый элемент (такт) MFCC массива это сумма с предыдущим элементом (тактом)
    # deg_path = np.sum(R_path, axis=1)
    # plt.plot(deg_path)
    # plt.title("deg_path")
    # plt.show()

    # PCP матрицу суммируем по x
    deg_rec = np.sum(Rf, axis=1)
    plt.plot(deg_rec)
    plt.title("deg_rec PCP")
    plt.show()

    # PCPandMFCC = deg_path + deg_rec
    # mu = deg_path.dot(PCPandMFCC) / np.sum((PCPandMFCC) ** 2)

    # PCPMatrixWithMU = mu * Rf
    # plt.imshow(PCPMatrixWithMU)
    # plt.title("PCPMatrixWithMU")
    # plt.show()

    # MFCCMatrixWithMU = (1 - mu) * R_path
    # plt.imshow(MFCCMatrixWithMU)
    # plt.title("MFCCMatrixWithMU")
    # plt.show()

    # A = PCPMatrixWithMU + MFCCMatrixWithMU

    # plt.imshow(A, aspect="auto")
    # plt.show()

    #####################################################
    # L = scipy.sparse.csgraph.laplacian(A, normed=True)
    # plt.imshow(L, aspect="auto")
    # plt.title("L")
    # plt.show()

    # evals, evecs = scipy.linalg.eigh(L)

    # plt.imshow(evecs, aspect="auto")
    # plt.title("evecs")
    # plt.show()

    # evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    # plt.imshow(evecs, aspect="auto")
    # plt.title("evecs median")
    # plt.show()

    # Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5
    # plt.imshow(Cnorm, aspect="auto")
    # plt.title("Cnorm")
    # plt.show()
    #
    k = 4
    # evecs_n = evecs[:, :k]
    # Cnorm_n = Cnorm[:, k - 1:k] + 1e-5
    #
    # plt.imshow(evecs_n, aspect="auto")
    # plt.title("evecs_n")
    # plt.show()
    #
    # plt.imshow(Cnorm_n, aspect="auto")
    # plt.title("Cnorm_n")
    # plt.show()
    #
    # X = evecs_n / Cnorm_n
    # plt.imshow(X, aspect="auto")
    # plt.title("X")
    # plt.show()

    # evecs_sum = np.sum(Rf, axis=1)
    mfcc_t = mfcc.T
    deg = np.array([deg_rec]).T

    frame_times = librosa.frames_to_time(beats, sr=sr)[:-1]
    # tick_hms = np.array([str(time.strftime("%M.%S", time.gmtime(s))) for s in frame_times])
    beat_times = np.array([float('{:.1f}'.format(s)) for s in frame_times])
    # beat_times = np.round(frame_times, 2)
    beat_times = np.array([beat_times]).T.astype(str)

    # conc = np.concatenate((mfcc_t, deg, beat_times), axis=1)
    # np.savetxt("foo.csv", conc, delimiter=",", fmt="%s")

    # conc = np.concatenate((mfcc_t, deg), axis=1)
    #
    # df = pd.read_csv('12.csv', delimiter=';', header=None)
    # X = df.iloc[:, :-1]
    # Y = df.iloc[:, -1]
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    #
    # classifier = LogisticRegression(solver='lbfgs', random_state=0)
    # classifier.fit(X_train, Y_train)
    # seg_ids = classifier.predict(conc)
    # print('Accuracy: {:.2f}'.format(classifier.score(X_test, Y_test)))

    # conc = np.stack((deg_rec, mfcc)).T
    # KM = sklearn.cluster.KMeans(n_clusters=4, n_init=50, max_iter=500)
    # seg_ids = KM.fit_predict(conc)

    seg = np.copy(seg_ids)
    plt.plot(seg)
    plt.show()

    count = 0
    dif_index = 0
    for idx, item in enumerate(seg_ids):
        if len(seg_ids) != idx:
            if seg_ids[idx] != seg_ids[idx - 1]:
                if count < 16:
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
    bound_idxs = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    bound_segs = list(seg_ids[bound_idxs])

    # Tack on the end-time
    # bound_idxs = list(np.append(bound_idxs, len(Cnorm) - 1))


    est_times = np.concatenate(([0], frame_times[bound_idxs]))
    silence_label = np.max(bound_segs) + 1
    est_labels = np.concatenate(([silence_label], bound_segs, [silence_label]))

    # # Remove empty segments if needed
    # est_times, est_labels = remove_empty_segments(est_times, est_labels)

    # second to milliseconds
    est_times = (est_times * 1000).astype(int)
    create_serato_markers(est_times[:5], est_labels[:5], audio_data)
    sys.exit()