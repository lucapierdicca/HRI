import os, sys, time, cv2, exercise

pdir = os.getenv('PNP_HOME')
sys.path.insert(0, pdir+'/PNPnaoqi/py')

import pnp_cmd_naoqi
from pnp_cmd_naoqi import *



# callbacks------------------------------------------------
def onfacedetected(value):
	global p,leds_service

	if value == []: whiteEyes()
	else:
		if p.get_condition('notfirstfacedetected') == False:
			p.set_condition('notfirstfacedetected',True)
		cyanEyes()

#-----------------------------------------------------------


# behaviour settings---------------------------------------
def basicAwareness():
	global ba_service
	
	stimulus = ['Touch',
				'TabletTouch',
				'Movement',
				'NavigationMotion']
	
	for i in stimulus:
		ba_service.setStimulusDetectionEnabled(i,False)

	ba_service.setStimulusDetectionEnabled('People',True)
	ba_service.setStimulusDetectionEnabled('Sound',True)

	ba_service.setEnabled(True)


def whiteEyes():
	global leds_service
	leds_service.on("FaceLeds")


def cyanEyes():
	global leds_service
	leds_service.off("LeftFaceLedsGreen")
	leds_service.off("RightFaceLedsGreen")

#-------------------------------------------------------------


NAME = 'Luca_Pierdicca'


p = PNPCmd()

p.begin()


# opening services
memory_service  = p.app.session.service("ALMemory")
fd_service = p.app.session.service("ALFaceDetection")
asr_service = p.app.session.service("ALSpeechRecognition")
leds_service = p.app.session.service("ALLeds")
ba_service = p.app.session.service("ALBasicAwareness")


input('Start...')
time.sleep(5)

# face
fdsuber = memory_service.subscriber("FaceDetected")
fdsign = fdsuber.signal.connect(onfacedetected)
fd_service.subscribe("face")



#--------------------------INTERACTION--------------------------
p.set_condition('notfirstfacedetected',False)




p.exec_action('Wait',interrupt='notfirstfacedetected')

#if prima_visita:
	# pepper: chiede il nome
	# pepper: saluta
	# pepper: spiega l'esercizio e le intenzioni  
#else:
	# pepper: chiede informazione salute
	# pepper: pronuncia riepilogo ultima visita, numero visite

p.exec_action('Show', '')
p.exec_action('Say',"Ok,_let's_get_started.")
p.exec_action('Say',"It's_easy,_put_the_magic_cube_on_the_start_cell_and_tell_me_when_you_are_ready.")
p.exec_action('Say',"When_you_think_you're_done,_call_me_and_I_will_display_your_draw_on_the_screen.")

drawn_moveslbl = []
exercise.reset_imgs()
sampled_moveslbl = exercise.pepper_img()
p.exec_action('Show', '')

	
for i in range(4):
	print('Start capture')
	cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280) 
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
	curr_trajectory = exercise.trajectory(cap)
	cap.release()
	print('End capture')

	# trajectory -> move_dict: '01':[(),(),...()]
	move_dict = exercise.discretize(curr_trajectory)
	# move_dict -> nn_movelbl: nearest_neighbor(move_dict.keys(), moves)
	nn_movelbl = exercise.nn(move_dict)
	drawn_moveslbl.append(nn_movelbl)
	# draw the nearest_neighbor
	img_move = exercise.render(moves[nn_move_lbl])
	
	cv2.imwrite('.\\tablet\\'+str(i+1)+'_c.png', img_move)
	
	p.exec_action('Show', '')

p.exec_action('Say','Ok,the_exercise_is_over._Good_bye!')


fd_service.unsubscribe("test_face")
fdsuber.signal.disconnect(fdsign)
	


p.end()