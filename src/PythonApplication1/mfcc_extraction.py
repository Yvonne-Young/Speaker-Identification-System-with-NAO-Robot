#==============================================================================#
#file: mfcc_extraction.py
#author: Yifan Yang
#updated: 6/17/2019
#==============================================================================#

from utility import *

#pre_processed: vector that after pre-process
#N: N-point FFT
#M: number of triangular filters
#L: L-point DCT
#sampling_rate for self-recording is 441000
#sampling_rate for TIMIT is 16000
def mfcc_extraction(pre_processed, N, M,L):
    num_of_frames = pre_processed.shape[0]   #985 for test
    points_per_frame = pre_processed.shape[1] #882 for test
    mfcc = np.zeros((num_of_frames,L))
    feature = np.zeros((num_of_frames,L*3))
    for i in range(num_of_frames):
        magnitude = abs(np.fft.fft(pre_processed[i,:],N))
        mel_log = mel_filter(magnitude, 16000, 30, points_per_frame)
        mfcc[i,:] = scipy.fftpack.dct(mel_log, n=L)
        mfcc_1 = derivation(mfcc[i,:])
        mfcc_2 = derivation(mfcc_1)
        tmp = np.hstack((mfcc[i,:],mfcc_1))
        feature[i,:] = np.hstack((tmp,mfcc_2))

    return feature
