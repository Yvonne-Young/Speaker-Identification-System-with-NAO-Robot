#==============================================================================#
#file: pre_processing.py
#author: Yifan Yang
#updated: 6/17/2019
#==============================================================================#

import numpy as np
import wave
import struct
#import matplotlib.pyplot as plt
from utility import * 

#pre-processing includes: sampling -> high pass filter -> framing & windowing
def pre_processing(file_name):
    channel = file_name.getnchannels()
    sampling_rate = file_name.getframerate()
    sample_width = file_name.getsampwidth()     #2 bytes
    num_points = file_name.getnframes()
    data = file_name.readframes(num_points)
    data = struct.unpack('{n}h'.format(n=num_points*channel),data)
    data = np.array(data)
     
    #high pass filter
    data_hpf = HPF(data,10000)

    #framing & windowing
    #20ms/frame, overlap = 10ms, 986 frames in total for test
    frame_size = sampling_rate*20/1000
    overlap = frame_size/2
    num_of_frames = data.shape[0] / (frame_size - overlap) - 1
    frame_matrix = np.zeros((num_of_frames,frame_size))
    for i in range(num_of_frames):
        for j in range(frame_size):
            frame_matrix[i,j] = data_hpf[i*(frame_size-overlap)+j]
        frame_matrix[i,:] = hamming_window(frame_matrix[i,:], frame_size, 0.46)

    return frame_matrix
