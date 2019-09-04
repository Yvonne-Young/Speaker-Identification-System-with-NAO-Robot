#==============================================================================#
#file: top.py
#author: Yifan Yang
#updated: 6/17/2019
#==============================================================================#

import wave
import os
import string 
import time 
from speaker import Speaker
from pre_processing import *
from mfcc_extraction import *

def training():     #return a dictionary of speakers' GMM model
    #TRAINING STEP
    #1. get all wave files under TIMIT data set
    data_set = os.listdir("F:\EE\graduation project\dataset\Yvonne")
    py_result = os.listdir("F:\EE\graduation project\python result\GMM model")
    speaker_list = []
    speaker_dic = {}

    for name in data_set:
        n = name + "_covars.txt"
        if n in py_result:
            pass
        else:
            sub_dir = "F:\EE\graduation project\dataset\Yvonne\%s"%name
            speaker_i = Speaker(name,sub_dir)
            print name
            gmm = speaker_i.get_GMM()
            print "done"
            save_model(gmm.weights,gmm.means,gmm.covars,name)
            speaker_list.append(speaker_i)
            speaker_dic[name] = speaker_i
                                                                                                                                                                                                                                                         
    #2. get GMM models for all speakers
    speaker_list = []
    speaker_dic = {}
    
    for wf in wave_list:
        fp = wave.open("F:\EE\graduation project\dataset\Yvonne\%s.wav"%wf,'r')
        speaker_i = Speaker(wf,fp)
        print wf 
        gmm = speaker_i.get_GMM()   #why cannot mfcc and gmm both be run
        print "done"
        save_model(gmm.weights,gmm.means,gmm.covars,wf)
        fp.close()
        speaker_list.append(speaker_i)
        speaker_dic[wf] = speaker_i 

    return speaker_dic 


if __name__ == '__main__':
    #speaker_dic = training()
    fp = wave.open("F:\EE\graduation project\dataset\Yvonne\FCJF0.wav",'r')
    pre_processed = pre_processing(fp) 
    utter_mfcc = mfcc_extraction(pre_processed,512,30,13)
    utter_mfcc = Normalizer().fit_transform(utter_mfcc)
    fp.close()

    speaker_score = {}      
    scores = []
    data_set = os.listdir("F:\EE\graduation project\dataset\Yvonne")
    #get all speaker's name
    wave_list = []
    for name in data_set:
        if os.path.splitext(name)[1] == ".WAV":
            wave_list.append(name[:-4])
        else:
            continue

    for name in wave_list:
        GMM_model_weights = "F:\EE\graduation project\python result\GMM model\TIMIT_2\%s_weights.txt"%name
        GMM_model_means   = "F:\EE\graduation project\python result\GMM model\TIMIT_2\%s_means.txt"%name
        GMM_model_covars  = "F:\EE\graduation project\python result\GMM model\TIMIT_2\%s_covars.txt"%name
        (weights,means,covars) = load_model(GMM_model_weights,GMM_model_means,GMM_model_covars,5,39) 
        prediction = scoring(utter_mfcc,weights,means,covars)
        scores.append(prediction)
        speaker_score[name] = prediction
        print ("the score of speaker " + name + " is " + str(prediction))

    predicted_speaker = max(speaker_score,key = speaker_score.get)
    print predicted_speaker
