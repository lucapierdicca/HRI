import qi
import argparse
import sys
import os
from pprint import pprint


# ALMemory
# subscriber()
# subscribeToEvent()
# unsubscribeToEvent()

# ALExtractor
# subscribe()
# unsubscribe()

# List of modules inheriting from ALExtractor
# ALCloseObjectDetection
# ALEngagementZones
# ALFaceDetection
# ALGazeAnalysis
# ALLandMarkDetection
# ALPeoplePerception
# ALRedBallDetection
# ALSittingPeopleDetection
# ALSonar
# ALVisualSpaceHistory
# ALWavingDetection
# ALVisionRecognition
# ALSegmentation3D +


def onWordRecognized(value):
    print ("value",value)


def onFaceDetected(value):
	print('Face')
	#print ("value",value)


def eventsInfo(memory_service):
	e_list = memory_service.getEventList()
	pprint(e_list)

	for e in e_list:
		print(memory_service.getSubscribers(e))



    
pip = ''
pport = 0




#Starting application
try:
    connection_url = "tcp://" + pip + ":" + str(pport)
    app = qi.Application(["easytest", "--qi-url=" + connection_url ])
except RuntimeError:
    print ("Can't connect to Naoqi at ip \"" + pip + "\" on port " + str(pport) +".\n"
           "Please check your script arguments. Run with -h option for help.")
    sys.exit(1)

app.start()
session = app.session



# starting services
memory_service  = session.service("ALMemory")
fd_service = session.service("ALFaceDetection")
asr_service = session.service("ALSpeechRecognition")
tts_service = session.service("ALTextToSpeech")

# events info
eventsInfo(memory_service)

# disable face tracking
fd_service.setTrackingEnabled("False")



# subscriber & signal+callback
fdsuber = memory_service.subscriber("FaceDetected")
fdsuber.signal.connect(onFaceDetected)
# start capturing & writing in memory in FaceDetected
fd_service.subscribe("test_face")


#let it run
app.run()


#Disconnecting callbacks and subscribers
fd_service.unsubscribe("test_face")
fdsuber.signal.disconnect(onFaceDetected)
    

