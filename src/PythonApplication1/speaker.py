#==============================================================================#
#file: speaker.py
#author: Yifan Yang
#updated: 6/17/2019
#==============================================================================#

from pre_processing import *
from mfcc_extraction import *
from utility import *

class Speaker(object):
    def __init__(self, speaker_id, wave_file,feature = None,model = None):
        self.speaker_id = speaker_id
        self.wave_file = wave_file
        self.feature = feature
        self.model = model 
      
    def get_mfcc(self):
        mfcc = []
        waves = os.listdir(self.wave_file)
        for file in waves:
            fp = wave.open("F:\EE\graduation project\dataset\BiCASLer\%s\%s"%(self.speaker_id,file),'r')
            pre_processed = pre_processing(fp)
            mfcc_file = mfcc_extraction(pre_processed,512,30,13)
            mfcc.extend(mfcc_file)
            fp.close()
        self.feature = mfcc 
        print np.shape(mfcc)
        return mfcc 

    def get_GMM(self):
        mfcc = self.get_mfcc()
        K = 5
        mfcc = Normalizer().fit_transform(mfcc)
        gmm = GMM(mfcc, K)
        print "building GMM model..."
        gmm.GMM_EM()
        self.model = gmm   
        return gmm 
