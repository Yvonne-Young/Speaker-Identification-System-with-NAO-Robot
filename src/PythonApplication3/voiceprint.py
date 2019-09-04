#==============================================================================#
#file: voiceprint.py
#author: Yifan Yang
#updated: 6/17/2019
#==============================================================================#

import socket
import librosa
import numpy as np
from mfcc_extraction import *
from speaker import *
import wave
import string 
from naoqi import ALProxy
import time
import string
import struct
from sklearn.externals import joblib

HOST = ""  
PORT = 21588
BUFSIZ = 57600000 
ADDR = (HOST, PORT)
robot_IP = '192.168.1.101'

#connect with NAO robot
tcpCliSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
tcpCliSock.connect((robot_IP,PORT))
time.sleep(2)

def training():
    dataset = os.listdir("C:\Users\win\Desktop\NAO_dataset\BiCASLer")
    speaker_list = []
    speaker_dic = {}

    for name in dataset:
        sub_dir = "C:\Users\win\Desktop\NAO_dataset\BiCASLer\%s"%name
        speaker_i = Speaker(name,sub_dir)
        print name
        gmm = speaker_i.get_GMM()
        joblib.dump(gmm,'%s.model'%name)
        print "done"
        speaker_list.append(speaker_i)
        speaker_dic[name] = speaker_i
    return speaker_dic

#get utterance data from NAO robot IP
#use tcp transaction
while True:  
    start = time.clock()
    i=1
    string_list = []
    while True:
        tcpCliSock.send("Conneted! Start processing...Please wait for a minute!")
        list_size = int(tcpCliSock.recv(6))
        print list_size
        data = [' ' for k in range(list_size)]

        #receive 10 bytes/time
        for k in range(list_size):
            data[k] = float(tcpCliSock.recv(10))

        sr = 16000
        y = np.array(data)
        hop_length = 512
        mfcc = librosa.feature.mfcc(y=y,sr=sr,hop_length=hop_length,n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
        feature = np.vstack([mfcc,mfcc_delta,mfcc_delta_delta])
        feature = feature.T
                
        utter_feature = Normalizer().fit_transform(feature) 
        dataset = os.listdir("C:\Users\win\Desktop\NAO_dataset\BiCASLer\")
        scores = {}
        for name in dataset:
            estimator = joblib.load("%s.model"%name)
            score = estimator.score(utter_feature)
            scores[name] = score
            print name + ' ' + str(score)
        predicted_speaker = max(scores,key=scores.get)
        message = "The speaker is: "+ predicted_speaker
        tcpCliSock.send(message) 
        print "The speaker is: " + predicted_speaker
        end = time.clock()
        print "Run time: %s seconds"%(end-start)
        
        i-=1
        if i == 0:
            break
    break
    
tcpCliSock.close()
