#==============================================================================#
#file: socket.py
#author: Yifan Yang
#updated: 6/17/2019
#==============================================================================#

import numpy as np
import socket
from time import ctime
from naoqi import ALProxy
import os
import wave
import struct
import string

robot_IP='192.168.1.101'
robot_PORT=9559

client_IP = ''
HOST = '192.168.26.1'
PORT = 21588
BUFSIZ = 1024
ADDR = (HOST, PORT)
connection = None
CONNECT = False

tcpSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpSerSock.bind((robot_IP,PORT))
tcpSerSock.listen(10)

record_path = "/home/nao/utterance.wav"


def pre_processing(file_name):
    channel = file_name.getnchannels()
    sampling_rate = file_name.getframerate()
    sample_width = file_name.getsampwidth()     #2 bytes
    num_points = file_name.getnframes()
    data = file_name.readframes(num_points)
    data = struct.unpack('{n}h'.format(n=num_points*channel),data)
    data = np.array(data)
    return data


class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self)
        self.tts = ALProxy('ALTextToSpeech')

    def onLoad(self):
        #put initialization code here
        pass

    def onUnload(self):
        #put clean-up code here
        pass

    def onInput_onStart(self):
        #self.onStopped() #activate the output of the box
        fp = wave.open(record_path,'r')
        y = pre_processing(fp)
        print np.shape(y)
        size = np.shape(y)[0]
        y_string = [' ' for k in range(size)]
        count = 0
        for k in range(size):
            count += 1
            y_string[k] = '%10d'%y[k]
        
		#send utterance data to computer
		try:
            while True:
                global connection
                connection,address = tcpSerSock.accept()
                CONNECT = True
                while CONNECT == True:
                    try:
                        speech = connection.recv(1024)
                        self.tts.say(speech)
                        print size
                        connection.send(str(size))
                        time.sleep(2)

                        count = 1
                        for k in range(size):
                            count += 1
                            connection.send(y_string[k])
                    except socket.timeout:
                        print 'time out'
                predicted_speaker = connection.recv(1024)
                self.tts.say(predicted_speaker)
                connection.close()
        except KeyboardInterrupt:
           print "interrupted by user"
           sys.exit(0)

        pass

    def onInput_onStop(self):
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box