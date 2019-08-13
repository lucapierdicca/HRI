#!/usr/bin/env python

# animations
# http://doc.aldebaran.com/2-5/naoqi/motion/alanimationplayer-advanced.html#animationplayer-list-behaviors-pepper

# How to use

# export PEPPER_IP=<...>
# python
# >>> import pepper_cmd
# >>> from pepper_cmd import *
# >>> begin()
# >>> pepper_cmd.robot.<fn>()

import time
import os
import socket
import threading
import math
import random
import qi

from naoqi import ALProxy

# Python Image Library
import Image
from patients import *


app = None
session = None
tts_service = None
memory_service = None
motion_service = None
anspeech_service = None
tablet_service = None

robot = None        # PepperRobot object

RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"

# Sensors
headTouch = 0.0
handTouch = [0.0, 0.0] # left, right
sonar = [0.0, 0.0] # front, back


# Sensors

def sensorThread(robot):
    sonarValues = ["Device/SubDeviceList/Platform/Front/Sonar/Sensor/Value",
                  "Device/SubDeviceList/Platform/Back/Sonar/Sensor/Value"]
    headTouchValue = "Device/SubDeviceList/Head/Touch/Middle/Sensor/Value"
    handTouchValues = [ "Device/SubDeviceList/LHand/Touch/Back/Sensor/Value",
                   "Device/SubDeviceList/RHand/Touch/Back/Sensor/Value" ]

    t = threading.currentThread()
    while getattr(t, "do_run", True):
        robot.headTouch = robot.memory_service.getData(headTouchValue)
        robot.handTouch = robot.memory_service.getListData(handTouchValues)
        robot.sonar = robot.memory_service.getListData(sonarValues)
        #print "Head touch middle value=", robot.headTouch
        #print "Hand touch middle value=", robot.handTouch
        #print "Sonar [Front, Back]", robot.sonar
        time.sleep(1)
    #print "Exiting Thread"





def touchcb(value):
    print "value=",value

    touched_bodies = []
    for p in value:
        if p[1]:
            touched_bodies.append(p[0])

    print touched_bodies

asr_word = ''
asr_confidence = 0
asr_timestamp = 0



def sensorvalue(sensorname):
    global robot
    if (robot!=None):
        return robot.sensorvalue(sensorname)



# Begin/end

def begin():
    global robot
    print 'begin'
    if (robot==None):
        robot=PepperRobot()
        robot.connect()
    robot.begin()

def end():
    global robot
    print 'end'
    time.sleep(0.5) # make sure stuff ends
    if (robot!=None):
        robot.quit()


# Robot motion

def stop():
    global robot
    if (robot==None):
        begin()
    robot.stop()

def forward(r=1):
    global robot
    if (robot==None):
        begin()
    robot.forward(r)

def backward(r=1):
    global robot
    if (robot==None):
        begin()
    robot.backward(r)

def left(r=1):
    global robot
    if (robot==None):
        begin()
    robot.left(r)

def right(r=1):
    global robot
    if (robot==None):
        begin()
    robot.right(r)

def robot_stop_request(): # stop until next begin()
    if (robot!=None):
        robot.stop_request = True
        robot.stop()
        print("stop request")



# Wait

def wait(r=1):
    print 'wait',r
    for i in range(0,r):
        time.sleep(3)


# Sounds

def bip(r=1):
    print 'bip'


def bop(r=1):
    print 'bop'


# Speech

def say(strsay):
    global robot
    print 'Say ',strsay
    if (robot==None):
        begin()
    robot.say(strsay)

def asay(strsay):
    global robot
    print 'Animated Say ',strsay
    if (robot==None):
        begin()
    robot.asay(strsay)



# Other 

# Alive behaviors
def setAlive(alive):
    global robot
    robot.setAlive(alive)

def stand():
    global robot
    robot.stand()

def disabled():
    global robot
    robot.disabled()

def interact():
    global robot
    robot.interactive()


def showurl(url):
    global robot
    if (robot!=None):
        return robot.showurl(url)


def run_behavior(bname):
    global session
    beh_service = session.service("ALBehaviorManager")
    beh_service.startBehavior(bname)
    #time.sleep(10)
    #beh_service.stopBehavior(bname)


def takephoto():
    global robot
#    global session, tts_service
#    str = 'Take photo'
#    print(str)
#    #tts_service.say(str)
#    bname = 'takepicture-61492b/behavior_1'
#    run_behavior(bname)
    robot.takephoto()


def opendiag():
    global robot
    robot.introduction()

def sax():
        global robot
#    global session, tts_service
#    str = 'demo'
#    print(str)
#    bname = 'saxophone-0635af/behavior_1'
#    run_behavior(bname)
        robot.sax()


class PepperRobot:

    def __init__(self):
        self.isConnected = False
        # Sensors
        self.headTouch = 0.0
        self.handTouch = [0.0, 0.0] # left, right
        self.sonar = [0.0, 0.0] # front, back
        self.language = "English"
        self.stop_request = False
        self.frame_grabber = False
        self.face_detection = False
        self.got_face = False
        self.sth = None
        self.jointNames = ["HeadYaw", "HeadPitch",
               "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
               "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]

    # Connect to the robot
    def connect(self, pip=os.environ['PEPPER_IP'], pport=9559, alive=False):

        self.ip = pip
        self.port = pport
        if (self.isConnected):
            print("Robot already connnected.")
            return

        print("Connecting to robot %s:%d ..." %(pip,pport))
        try:
            connection_url = "tcp://" + pip + ":" + str(pport)
            self.app = qi.Application(["Pepper command", "--qi-url=" + connection_url ])
            self.app.start()
        except RuntimeError:
            print("%sCannot connect to Naoqi at %s:%d %s" %(RED,pip,pport,RESET))
            self.session = None
            return

        print("%sConnected to robot %s:%d %s" %(GREEN,pip,pport,RESET))
        self.session = self.app.session

        print("Starting services...")

        #Starting services
        self.memory_service  = self.session.service("ALMemory")
        self.motion_service  = self.session.service("ALMotion")
        self.tts_service = self.session.service("ALTextToSpeech")
        self.anspeech_service = self.session.service("ALAnimatedSpeech")
        self.leds_service = self.session.service("ALLeds")
        self.asr_service = None
        self.tablet_service = self.session.service("ALTabletService")
        self.fd_service = self.session.service("ALFaceDetection")
        try:
            
            self.animation_player_service = self.session.service("ALAnimationPlayer")
            self.beh_service = self.session.service("ALBehaviorManager")
            self.al_service = self.session.service("ALAutonomousLife")
            self.rp_service = self.session.service("ALRobotPosture")
            self.bm_service = self.session.service("ALBackgroundMovement")
            self.ba_service = self.session.service("ALBasicAwareness")
            self.sm_service = self.session.service("ALSpeakingMovement")
            self.asr_service = self.session.service("ALSpeechRecognition")
            self.face_service = self.session.service("ALFaceDetection")

            self.alive = alive
            print('Alive behaviors: %r' %self.alive)

            self.bm_service.setEnabled(self.alive)
            self.ba_service.setEnabled(self.alive)
            self.sm_service.setEnabled(self.alive)
            
            #webview = "http://198.18.0.1/apps/spqrel/index.html"
            #self.tablet_service.showWebview(webview)

            self.touch_service = self.session.service("ALTouch")
            self.touchstatus = self.touch_service.getStatus()
            #print touchstatus
            self.touchsensorlist = self.touch_service.getSensorList()
            #print touchsensorlist

        except:
            print "Services not available."

        #anyTouch = self.memory_service.subscriber("TouchChanged")
        #idAnyTouch = anyTouch.signal.connect(touchcb)

        # create a thead that monitors directly the signal
        self.sth = threading.Thread(target = sensorThread, args = (self, ))
        self.sth.start()

        self.isConnected = True


    def quit(self):
        print "Quit Pepper robot."
        if self.sth != None:
            self.sth.do_run = False
        time.sleep(1)
        self.app.stop()

    # general commands

    def begin(self):
        self.stop_request = False

    def exec_cmd(self, params):
        cmdstr = "self."+params
        print "Executing %s" %(cmdstr)
        eval(cmdstr)    


    # Leds---------------------------------------------------------------
    def whiteEyes(self):
        self.leds_service.on('FaceLeds')

    def greenEyes(self):
        self.leds_service.off('LeftFaceLedsRed')
        self.leds_service.off('LeftFaceLedsBlue')
        self.leds_service.off('RightFaceLedsRed')
        self.leds_service.off('RightFaceLedsBlue')

    def cyanEyes(self):
        self.leds_service.off("LeftFaceLedsRed")
        self.leds_service.off("RightFaceLedsRed")



    # Face------------------------------------------------------------
    def onFaceDetected(value):
        self.cyanEyes()
        self.got_face = True

    def startFaceDetection(self):
        self.fdsuber = self.memory_service.subscriber("FaceDetected")
        self.fdsign = self.fdsuber.signal.connect(self.onFaceDetected)
        self.fd_service.subscribe("face")

    def stopFaceDetection(self):
        self.whiteEyes()
        self.fd_service.unsubscribe("face")
        self.fdsuber.signal.disconnect(self.fdsign)



    # Face edo------------------------------------------------------
    def startFaceDetection_(self):

        if self.face_detection:  # already active
            return

        # Connect to camera
        #self.startFrameGrabber()

        # Connect the event callback.
        self.frsub = self.memory_service.subscriber("FaceDetected")
        self.ch1 = self.frsub.signal.connect(self.on_facedetected_)
        self.got_face = False
        self.savedfaces = []
        self.face_detection = True
        self.face_recording = False # if images are saved on file
        self.first_time = True


    def stopFaceDetection_(self):
        self.frsub.signal.disconnect(self.ch1)
        self.camProxy.unsubscribe(self.videoClient)
        self.face_recording = False
        self.face_detection = False

    check = False
    def on_facedetected_(self, value):
        self.faceLabel = ''
        
        """
        Callback for event FaceDetected.
        """
        faceID = -1

        if value == []:  # empty value when the face disappears
            self.got_face = False
            self.whiteEyes()
        elif not self.got_face:  # only speak the first time a face appears
            self.got_face = True
            self.cyanEyes()
            #print "I saw a face!"
            #self.tts.say("Hello, you!")
            # First Field = TimeStamp.
            timeStamp = value[0]
            #print "TimeStamp is: " + str(timeStamp)

            # Second Field = array of face_Info's.
            faceInfoArray = value[1]
 
            for j in range( len(faceInfoArray)-1 ):
                faceInfo = faceInfoArray[j]


                # First Field = Shape info.
                faceShapeInfo = faceInfo[0]

                # Second Field = Extra info (empty for now).
                faceExtraInfo = faceInfo[1]

                faceID = faceExtraInfo[0]

                scoreReco = faceExtraInfo[1]
                faceLabel = faceExtraInfo[2]
                print('ScoreReco:', scoreReco)
                print('faceLabel:', faceLabel)


                # print "Face Infos :  alpha %.3f - beta %.3f" % (faceShapeInfo[1], faceShapeInfo[2])
                # print "Face Infos :  width %.3f - height %.3f" % (faceShapeInfo[3], faceShapeInfo[4])
                # print "Face Extra Infos :" + str(faceExtraInfo)

                print "Face ID: %d" %faceID
            time.sleep(2)
            self.patient = faceLabel
            
            if self.first_time:
                self.first_time = False

                try:
                    if faceLabel != "":
                        self.patient = faceLabel
                        #self.old_Patient()
                        self.old_patient = True
                        self.check = True
                        

                    else:
                        #self.new_Patient()
                        self.old_patient = False
                        self.check = True

                except RuntimeError:
                    print("try-except new-old patient")
                    end()

            #print faceInfoArray
            print self.face_service.getLearnedFacesList()
        # if self.camProxy!=None and faceID>=0 and faceID not in self.savedfaces and self.face_recording:
        #     # Get the image 
        #     img = self.camProxy.getImageRemote(self.videoClient)

        #     # Get the image size and pixel array.
        #     imageWidth = img[0]
        #     imageHeight = img[1]
        #     array = img[6]

        #     # Create a PIL Image from our pixel array.
        #     im = Image.frombytes("RGB", (imageWidth, imageHeight), array)

        #     # Save the image.
        #     fname = "face_%03d.png" %faceID
        #     im.save(fname, "PNG")
        #     print "Image face %d saved." %faceID

        #     self.savedfaces.append(faceID)





    # Awareness---------------------------------------------------------
    def startBasicAwareness(self):
        stimulus = ['Touch',
                    'TabletTouch',
                    'Movement',
                    'NavigationMotion']
        
        for i in stimulus:
            self.ba_service.setStimulusDetectionEnabled(i,False)

        self.ba_service.setStimulusDetectionEnabled('People',True)
        self.ba_service.setStimulusDetectionEnabled('Sound',True)
        self.ba_service.setEnabled(True)

    def stopBasicAwareness(self):
        #self.normalPosture()
        self.ba_service.setEnabled(False)




    # Tablet------------------------------------------------------------
    def show(self, url):
        self.tablet_service.showWebview(url)
        #self.tablet_service.reloadPage(True)




    # ASR----------------------------------------------------------------
    def onWordRecognized(value):
        global asr_word, asr_confidence, asr_timestamp
        
        print "ASR value = ",value,time.time()
        if (value[1]>0):
            asr_word = value[0]
            asr_confidence = value[1]
            asr_timestamp = time.time()


    def ask(self, sentence, vocabulary, timeout=5):
        global asr_word, asr_confidence, asr_timestamp

        self.asay2(sentence)

        if (self.asr_service is None):
            return ''
        self.asr_service.pause(True)
        self.asr_service.setVocabulary(vocabulary, True)
        self.asr_service.pause(False)

        asr_word = ''
        while asr_word == '':
            self.asr_service.subscribe("asr_pepper_cmd")

            subWordRecognized = self.memory_service.subscriber("WordRecognized")
            idSubWordRecognized = subWordRecognized.signal.connect(self.onWordRecognized)

            i = 0
            dt = 0.5
            while (i<timeout and asr_word==''):
                time.sleep(dt)
                i += dt

            self.asr_service.unsubscribe("asr_pepper_cmd")
            subWordRecognized.signal.disconnect(idSubWordRecognized)

            dt = time.time() - asr_timestamp
            if (dt<timeout and asr_confidence>0.3):
                return asr_word[asr_word.find('> ')+2:asr_word.find(' <')]
            else:
                asr_word = ''
                time.sleep(0.5)




    # Speech------------------------------------------------------------------

    # English, Italian, French
    def setLanguage(self, lang):
        languages = {"en" : "English", "it": "Italian"}
        if  (lang in languages.keys()):
            lang = languages[lang]
        self.tts_service.setLanguage(lang)

    def say(self, interaction):
        if self.stop_request:
            return
        self.tts_service.setParameter("speed", 90)
        self.tts_service.say(interaction)
        #self.asay2(interaction)

    def asay2(self, interaction):
        if self.stop_request:
            return
        # set the local configuration
        configuration = {"bodyLanguageMode":"contextual"}
        self.anspeech_service.say(interaction, configuration)

    def asay(self, interaction):
        if self.stop_request:
            return
        # set the local configuration
        #configuration = {"bodyLanguageMode":"contextual"}

        # http://doc.aldebaran.com/2-5/naoqi/motion/alanimationplayer-advanced.html#animationplayer-list-behaviors-pepper
        vanim = ["animations/Stand/Gestures/Enthusiastic_4",
                 "animations/Stand/Gestures/Enthusiastic_5",
                 "animations/Stand/Gestures/Excited_1",
                 "animations/Stand/Gestures/Explain_1" ]
        anim = random.choice(vanim) # random animation

        if ('hello' in interaction):
            anim = "animations/Stand/Gestures/Hey_1"
    
        self.anspeech_service.say("^start("+anim+") " + interaction+" ^wait("+anim+")")








    
    # Alive behaviors-------------------------------------------------------------
    def setAlive(self, alive):
        self.alive = alive
        print('Alive behaviors: %r' %self.alive)
        self.bm_service.setEnabled(self.alive)
        self.ba_service.setEnabled(self.alive)
        self.sm_service.setEnabled(self.alive)



    def animation(self, interaction):
        if self.stop_request:
            return
        print 'Animation ',interaction
        self.bm_service.setEnabled(False)
        self.ba_service.setEnabled(False)
        self.sm_service.setEnabled(False)

        self.animation_player_service.run(interaction)

        self.bm_service.setEnabled(self.alive)
        self.ba_service.setEnabled(self.alive)
        self.sm_service.setEnabled(self.alive)

    


    
    # Robot motion-----------------------------------------------------------------
    def stop(self):
        print 'stop'
        self.motion_service.stopMove()
        bns = self.beh_service.getRunningBehaviors()
        for b in bns:
            self.beh_service.stopBehavior(b)

    def forward(self, r=1):
        if self.stop_request:
            return
        print 'forward',r
        #Move in its X direction
        x = r
        y = 0.0
        theta = 0.0
        self.motion_service.moveTo(x, y, theta) #blocking function

    def backward(self, r=1):
        if self.stop_request:
            return
        print 'backward',r
        x = -r
        y = 0.0
        theta = 0.0
        self.motion_service.moveTo(x, y, theta) #blocking function

    def left(self, r=1):
        if self.stop_request:
            return
        print 'left',r
        #Turn 90deg to the left
        x = 0.0
        y = 0.0
        theta = math.pi/2 * r
        self.motion_service.moveTo(x, y, theta) #blocking function

    def right(self, r=1):
        if self.stop_request:
            return
        print 'right',r
        #Turn 90deg to the right
        x = 0.0
        y = 0.0
        theta = -math.pi/2 * r
        self.motion_service.moveTo(x, y, theta) #blocking function

    

    # Head motion-----------------------------------------------------------------
    def headPose(self, yaw, pitch, tm):
        jointNames = ["HeadYaw", "HeadPitch"]
        initAngles = [yaw, pitch]
        timeLists  = [tm, tm]
        isAbsolute = True
        self.motion_service.angleInterpolation(jointNames, initAngles, timeLists, isAbsolute)


    def headscan(self):
        jointNames = ["HeadYaw", "HeadPitch"]

        # look left
        initAngles = [1.6, -0.2]
        timeLists  = [5.0, 5.0]
        isAbsolute = True
        self.motion_service.angleInterpolation(jointNames, initAngles, timeLists, isAbsolute)

        # look right
        finalAngles = [-1.6, -0.2]
        timeLists  = [10.0, 10.0]
        self.motion_service.angleInterpolation(jointNames, finalAngles, timeLists, isAbsolute)

        # look ahead center
        finalAngles = [0.0, -0.2]
        timeLists  = [5.0, 5.0]
        self.motion_service.angleInterpolation(jointNames, finalAngles, timeLists, isAbsolute)
        

    # Arms stiffness [0,1]
    def setArmsStiffness(self, stiff_arms):
        names = "LArm"
        stiffnessLists = stiff_arms
        timeLists = 1.0
        self.motion_service.stiffnessInterpolation(names, stiffnessLists, timeLists)

        names = "RArm"
        self.motion_service.stiffnessInterpolation(names, stiffnessLists, timeLists)



    # Wait
    def wait(self, r=1):
        print 'wait',r
        for i in range(0,r):
            time.sleep(3)

    # Sensors
    def sensorvalue(self, sensorname):
        if (sensorname == 'frontsonar'):
            return self.sonar[0]
        elif (sensorname == 'rearsonar'):
            return self.sonar[1]
        elif (sensorname == 'headtouch'):
            return self.headTouch
        elif (sensorname == 'lefthandtouch'):
            return self.handTouch[0]
        elif (sensorname == 'righthandtouch'):
            return self.handTouch[1]


    def sensorvaluestring(self):
        return '%f,%f,%d,%d,%d' %(self.sensorvalue('frontsonar'),self.sensorvalue('rearsonar'),self.sensorvalue('headtouch'),self.sensorvalue('lefthandtouch'),self.sensorvalue('righthandtouch'))



    # Postures
    def normalPosture(self):
        jointValues = [0.00, -0.21, 1.55, 0.13, -1.24, -0.52, 0.01, 1.56, -0.14, 1.22, 0.52, -0.01]
        isAbsolute = True
        self.motion_service.angleInterpolation(self.jointNames, jointValues, 3.0, isAbsolute)


    def setPosture(self, jointValues, sec=3.0):
        isAbsolute = True
        self.motion_service.angleInterpolation(self.jointNames, jointValues, sec, isAbsolute)

    def getPosture(self):
        useSensors = True
        pose = self.motion_service.getAngles(self.jointNames, useSensors)
        return pose



    def raiseArm(self, which='R'): # or 'R'/'L' for right/left arm
        if (which=='R'):
            jointNames = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
            jointValues = [ -1.0, -0.3, 1.22, 0.52, -1.08]
        else:
            jointNames = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
            jointValues = [ -1.0, 0.3, -1.22, -0.52, 1.08]

        isAbsolute = True
        self.motion_service.angleInterpolation(jointNames, jointValues, 3.0, isAbsolute)


    def stand(self):
        if self.al_service.getState()!='disabled':
            self.al_service.setState('disabled')
        self.rp_service.goToPosture("Stand",2.0)

    def disabled(self):
        #self.tts_service.say("Bye bye")
        self.al_service.setState('disabled')

    def interactive(self):
        #tts_service.say("Interactive")
        self.al_service.setState('interactive')







