import os, sys

pdir = os.getenv('PNP_HOME')
sys.path.insert(0, pdir+'/PNPnaoqi/actions/')

# this allows to implement functions easily
import action_base
from action_base import *

pdir = os.getenv('PEPPER_TOOLS_HOME')
sys.path.append(pdir+ '/cmd_server')

import pepper_cmd
from pepper_cmd import *

pdir = os.getenv('MODIM_HOME')
sys.path.append(pdir + '/src/GUI')

import ws_client
from ws_client import *



class Rwrist(NAOqiAction_Base):
    
    def __init__(self, actionName, session, robot):
        NAOqiAction_Base.__init__(self,actionName, session)
        self.robot = robot

    def actionThread_exec (self,params):
        jointNames = ["RWristYaw"]
        jointValues = [float(params)]
        isAbsolute = True
        self.robot.motion_service.angleInterpolation(jointNames, jointValues, 2.0, isAbsolute)

        # action end
        action_termination(self.actionName,params)



class Lwrist(NAOqiAction_Base):
    
    def __init__(self, actionName, session, robot):
        NAOqiAction_Base.__init__(self,actionName, session)
        self.robot = robot

    def actionThread_exec (self,params):
        jointNames = ["LWristYaw"]
        jointValues = [float(params)]
        isAbsolute = True
        self.robot.motion_service.angleInterpolation(jointNames, jointValues, 2.0, isAbsolute)

        # action end
        action_termination(self.actionName,params)


class Rarm(NAOqiAction_Base):
    
    def __init__(self, actionName, session, robot):
        NAOqiAction_Base.__init__(self,actionName, session)
        self.robot = robot

    def actionThread_exec(self, params):
        jointNames = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
        jointValues = [float(i) for i in params.split('_')]

        isAbsolute = True
        self.robot.motion_service.angleInterpolation(jointNames, jointValues, 2.0, isAbsolute)


        # action end
        action_termination(self.actionName,params)


class Larm(NAOqiAction_Base):
    
    def __init__(self, actionName, session, robot):
        NAOqiAction_Base.__init__(self,actionName, session)
        self.robot = robot

    def actionThread_exec(self, params):
        jointNames = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
        jointValues = [float(i) for i in params.split('_')]

        isAbsolute = True
        self.robot.motion_service.angleInterpolation(jointNames, jointValues, 2.0, isAbsolute)


        # action end
        action_termination(self.actionName,params)


class normPosture(NAOqiAction_Base):

    def __init__(self, actionName, session, robot):
        NAOqiAction_Base.__init__(self,actionName, session)
        self.robot = robot

    def actionThread_exec(self, params):
        jointNames = ["HeadYaw", "HeadPitch",
               "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
               "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
        jointValues = [0.00, -0.21, 1.55, 0.13, -1.24, -0.52, 0.01, 1.56, -0.14, 1.22, 0.52, -0.01]
        isAbsolute = True
        self.robot.motion_service.angleInterpolation(jointNames, jointValues, 2.0, isAbsolute)

        # action end
        action_termination(self.actionName,params)


class Wave(NAOqiAction_Base):

    def __init__(self, actionName, session, robot):
        NAOqiAction_Base.__init__(self,actionName, session)
        self.robot = robot

    def actionThread_exec(self, params):
        jointNames = ["RElbowYaw"]

        self.robot.motion_service.openHand('RHand')

        RElbowYaw = 1.30
        isAbsolute = True
        
        for i in range(8):
            if i%2==0:
                jointValues = [RElbowYaw+float(params)]
            else:
                jointValues = [RElbowYaw-float(params)]
            self.robot.motion_service.angleInterpolation(jointNames, jointValues, 0.5, isAbsolute)

        # action end
        action_termination(self.actionName,params)



class Wait(NAOqiAction_Base):

    def actionThread_exec (self, params):

        # action exec
        while (self.do_run): 
            print "Action "+self.actionName+" "+params+" exec..."
            time.sleep(1)
            

        # action end
        action_termination(self.actionName,params)


class Show(NAOqiAction_Base):

    def __init__(self, actionName, session, robot,mc):
        NAOqiAction_Base.__init__(self,actionName, session)
        self.robot = robot
        self.mc = mc

        #im.executeModality('TEXT_title',params)        

    def actionThread_exec(self, params):

        self.robot.tablet_service.showWebview("http://10.0.1.204:8000")

        # action end
        action_termination(self.actionName,params)


class Say(NAOqiAction_Base):
    
    def __init__(self, actionName, session, robot):
        NAOqiAction_Base.__init__(self,actionName, session)
        self.robot = robot

    def actionThread_exec (self,params):
        self.robot.tts_service.setLanguage('English')
        self.robot.tts_service.setParameter("speed", 80)
        self.robot.tts_service.say(params)

        # action end
        action_termination(self.actionName,params)


def initActions():

    pepper_cmd.begin()

    # obtain the app object that will handle the connection to the robot
    app = pepper_cmd.robot.app 


    # modim 
    mc = ModimWSClient()
    mc.setCmdServerAddr(pepper_cmd.robot.ip, 9101)
    #im = InteractionManager(None,pepper_cmd.robot)

    # and then we have to register the actions with a name
    Say('Say',app.session, pepper_cmd.robot)
    Wait('Wait', app.session)
    Show('Show',app.session,pepper_cmd.robot,mc)

    # then we return the app object
    return app




# Start action server
if __name__ == "__main__":

    print("Starting action server (CTRL+C to quit)")

    app = initActions()

    app.run()