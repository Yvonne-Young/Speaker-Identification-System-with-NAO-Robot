import numpy as np
import scipy 
from scipy import fftpack
import struct 
import matplotlib.pyplot as plt
from math import *
from GMM import *
import os 

#vector: vector to be filtered
#d: the threshold of HPF  
def HPF(vector,d):
    f = np.fft.fft(vector)
    fshift = np.fft.fftshift(f)
    def make_transform_vector(d):
        trans_vec = np.zeros(vector.shape[0])
        center_point = round(vector.shape[0]/2)
        for i in range(vector.shape[0]):
            distance = abs(i - center_point)
            if distance <= d:
                trans_vec[i] = 0
            else:
                trans_vec[i] = 1
        return trans_vec
    d_vector = make_transform_vector(d)
    new_vec = np.abs(np.fft.ifft(np.fft.ifftshift(fshift*d_vector)))
    return new_vec

def hamming_window(vector, frame_size, alpha):
    hamming = np.zeros((1,vector.shape[0]))
    for i in range(frame_size):
        hamming[0,i] = vector[i]*((1-alpha)-alpha*cos(2*pi*i/(frame_size-1)))
    return hamming

def lin2mel(lin_freq):
    mel = 2595*log10(1+lin_freq/700.0)
    return mel
 
def mel2lin(mel_freq):
    lin = 700*(10**(mel_freq/2595.0)-1)
    return lin

#vector:input frame to be filtered
#sampling_rate: sampling rate, 44100 for test
#M: number of triangular filters
#X: X points per frame
def mel_filter(vector,sampling_rate,M,X):
    mel_min = 0
    mel_max = lin2mel(sampling_rate/2)
    step = (mel_max-mel_min)/(M+1)
    center_points = []
    start_points = []
    end_points = []
    
    i = 0
    while i < mel_max:
        center = int(floor(mel2lin(i)*X/(sampling_rate/2)))
        center_points.append(center)
        start_points.append(center)
        end_points.append(center)
        i += step
    
    del start_points[M] 
    del center_points[0]
    del end_points[0:2]
    end_points.append(X-1)
    
    SUM = np.zeros(M)
    Sout = np.zeros(M)
    for i in range(M):
        filter_mag = []
        for j in range(start_points[i],center_points[i]):
            fm = (j-start_points[i])/(center_points[i]-start_points[i])
            SUM[i] += (fm*vector[j])
        for j in range(center_points[i],end_points[i]):
            fm = (end_points[i]-j)/(end_points[i]-center_points[i])
            SUM[i] += (fm*vector[j])
        Sout[i] = 2*log10(SUM[i])

    return Sout

def derivation(input):
    output = np.zeros(np.shape(input))
    output[0] = input[0]
    output[1] = input[1]
    output[11] = input[11]
    output[12] = input[12]
    for i in range(2,11):
        output[i] = ((input[i+1]-input[i-1])+(input[i+2]-input[i-2]))/10.0
    return output 

#x:input mfcc matrix that under verification
#weights: claimed speaker's GMM model weights
#means: claimed speaker's GMM model means
#covs: claimed speaker's GMM model covariances
def scoring(x,weights,means,covs):
    prob_sum = 0
    prob_tmp = 0
    for j in range(np.shape(weights)[0]):
        for i in range(np.shape(x)[0]):
            dim = np.shape(covs)[1]
            covdet = np.linalg.det(covs[j] + np.eye(dim) * 0.001)
            covinv = np.linalg.inv(covs[j] + np.eye(dim) * 0.001)
            xdiff = (x[i,:] - means[j]).reshape((1,dim))
            prob = 1.0/(np.power(np.power(2*np.pi,dim*0.5)*np.abs(covdet),0.5))*\
                np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
            prob_tmp += log10(prob) 
        prob_sum += prob_tmp * weights[j]
    return prob_sum

#weights: GMM model weights
#means: GMM model means
#covars: GMM model covariance matrices
#name: the model belonger's name
def save_model(weights,means,covars,name):
    #file1 = "F:\EE\graduation project\python result\GMM model\TIMIT_2\%s_weights.txt"%name
    #file2 = "F:\EE\graduation project\python result\GMM model\TIMIT_2\%s_means.txt"%name
    #file3 = "F:\EE\graduation project\python result\GMM model\TIMIT_2\%s_covars.txt"%name

    file1 = "F:\EE\graduation project\python result\GMM model\TMP\%s_weights.txt"%name
    file2 = "F:\EE\graduation project\python result\GMM model\TMP\%s_means.txt"%name
    file3 = "F:\EE\graduation project\python result\GMM model\TMP\%s_covars.txt"%name

    fp1 = open(file1,'w')
    fp2 = open(file2,'w')
    fp3 = open(file3,'w')

    for weight in weights:
        fp1.write(str(weight))
        fp1.write('\n')
    for i in range (np.shape(means)[0]):
        for j in range(np.shape(means)[1]):
            fp2.write(str(means[i][j]))
            fp2.write(' ')
        fp2.write('\n')
    for i in range(np.shape(covars)[0]):
        for j in range(np.shape(covars)[1]):
            for k in range(np.shape(covars)[2]):
                fp3.write(str(covars[i][j][k]))
                fp3.write(' ')
            fp3.write('\n')
        fp3.write('\n')

    fp1.close()
    fp2.close()
    fp3.close()

#file_weights: GMM model weights saved file
#file_means: GMM model means matrix saved file
#file_covars: GMM model covariance matrices saved file
#K: number of Gaussian models
#dim: dimension
def load_model(file_weights,file_means,file_covars,K,dim):
    fp1 = open(file_weights,'r')
    fp2 = open(file_means,'r')
    fp3 = open(file_covars,'r')

    weights = []
    means = []
    means_col = []
    covars = np.zeros((K,dim,dim))

    for i in range(dim):
        means_col.append(0)
    for i in range(K):
        means.append(means_col)
    
    means_lines = fp2.readlines()
    covars_lines = fp3.readlines()
    covars_lines.remove('\n')
    covars_lines.remove('\n')
    covars_lines.remove('\n')
    covars_lines.remove('\n')

    for line in fp1.readlines():
        weights.append(float(line))

    for i in range(K):
        means_list_i = means_lines[i].split(' ')
        for j in range(dim):
            means[i][j] = float(means_list_i[j])

    for i in range(K):
        for j in range(dim):
            row = covars_lines[i*13+j].split(' ')[0:-1]
            for m in row:
                m = float(m)
            covars[i,j] = row

    fp1.close()
    fp2.close()
    fp3.close()

    return (weights,means,covars)

#speaker_id: id of the speaker under test
#scores: list, the matching scores of all models
def draw_det(speaker_id,scores):
    up = np.max(scores)
    down = np.min(scores)
    pos_num = len(scores)//2

    x = []
    y = []
    dot_num = 100
    step = (up-down)/(dot_num+1)
    threshold = up
    size = len(scores)
    for i in range(dot_num):
        threshold -= step
        false_neg = 0
        false_pos = 0
        for d in range(size):
            if d < pos_num and scores[d] < threshold:
                false_pos += 1
            elif d > pos_num and scores[d] > threshold:
                false_neg += 1

        #print(threshold,end="\t")      print(,end='') only works in python 3.0+ version
        #print("false alarm: %f"%(false_pos/size),end="\t")
        #print("missing: %f"%(false_neg/size))
        print(threshold)
        print("false alarm: %f"%(false_pos/size))
        print("missing: %f"%(false_neg/size))
        x.append(false_pos/size)
        y.append(false_neg/size)

    x = np.array(x)
    y = np.array(y)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("DET curves")
    plt.xlabel("False alarm probability(in%)")
    plt.ylabel("Miss probability(in%)")
    ax1.scatter(x,y,c='r',marker='.')
    plt.legend('x1')
    plt.show()
