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
    print("value",value)


def onFaceDetected(value):
	global leds_service

	print("Face")
	if value == []:
		whiteEyes()
	else:
		cyanEyes()
		#say('I see you')

	#print ("value",value)


def eventsInfo(memory_service):
	e_list = memory_service.getEventList()
	pprint(e_list)

def whiteEyes():
	leds_service.on("FaceLeds")

def cyanEyes():
	leds_service.off("LeftFaceLedsGreen")
	leds_service.off("RightFaceLedsGreen")

def normalPosture():
	global motion_service

	jointNames = ["HeadYaw", "HeadPitch","LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
	"RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]

	jointValues = [0.00, -0.21, 1.55, 0.13, -1.24, -0.52, 0.01, 1.56, -0.14, 1.22, 0.52, -0.01]
	isAbsolute = True
	motion_service.angleInterpolation(jointNames, jointValues, 3.0, isAbsolute)

def say(param):
	global tts_service
	tts_service.say(param)

    
pip = '10.0.1.201'
pport = 9559


stimulus = ['Touch','TabletTouch','Movement','NavigationMotion']

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
leds_service = session.service("ALLeds")
ba_service = session.service("ALBasicAwareness")
motion_service = session.service("ALMotion")

# start basic awareness
for i in stimulus:
	ba_service.setStimulusDetectionEnabled(i,False)

ba_service.setStimulusDetectionEnabled('People',True)
ba_service.setStimulusDetectionEnabled('Sound',True)

ba_service.setEnabled(False)


print(fd_service.getLearnedFacesList())
fd_service.learnFace('Luca')

# subscriber & signal+callback
# start capturing & writing in memory in FaceDetected
fdsuber = memory_service.subscriber("FaceDetected")
fdsign = fdsuber.signal.connect(onFaceDetected)
# module subscriber
fd_service.subscribe("test_face")



#let it run
app.run()


# resetting everything
fd_service.unsubscribe("test_face")
fdsuber.signal.disconnect(fdsign)
whiteEyes()
ba_service.setEnabled(False)
normalPosture()
    

