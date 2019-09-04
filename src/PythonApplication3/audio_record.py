#==============================================================================#
#file: audio_record.py
#author: Yifan Yang
#updated: 6/17/2019
#==============================================================================#

robot_IP = '192.168.1.101'
robot_PORT = 9559

class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self)
        utterance = ALProxy("ALAudioRecorder", robot_IP, robot_PORT)

    def onLoad(self):
        #put initialization code here
        pass

    def onUnload(self):
        #put clean-up code here
        pass

	#record 13s
	#set start record mark and record file save path
    def onInput_onStart(self):
        #self.onStopped() #activate the output of the box
        print "Start recording..."
        print "=============================================\n"*100
        record_path = "/home/nao/utterance.wav"
        utterance = ALProxy("ALAudioRecorder", robot_IP, robot_PORT)
        utterance.stopMicrophonesRecording()
        utterance.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
        time.sleep(13)
        utterance.stopMicrophonesRecording()
        print "Record over"
        pass

    def onInput_onStop(self):
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box