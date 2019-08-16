import os, sys

import sar_cmd
from sar_cmd import *

import exercise
import pickle
from pprint import pprint

import cv2
import posture
import numpy

from statistics import mean


# default patient data
patient_data = {'personali':{'nome':'unk',
								  'durata':0,
								  'seduta':0},
				'sessioni':[],
				'perrore':{'left ear':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'right ear':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'left eye':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'right eye':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'head':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'nose':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'mouth':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'right shoulder':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'left shoulder':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
						  'belly':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]}
			   }

int_to_moveslbl = {i:k for i,(k,v) in enumerate(patient_data['perrore'].items())}
moveslbl_to_int = {v:k for k,v in int_to_moveslbl.items()}


server = "http://10.0.1.203:8000"
nome = 'Luca_Pierdicca'
durata = 5
massa = 0.1
ripetizioni = 3
device = 0


# fetching patient data
if len(os.listdir('./'+nome)) == 0:
	pickle.dump(patient_data,open('./'+nome+'/'+nome+'.pickle','w'))
else:
	patient_data = pickle.load(open('./'+nome+'/'+nome+'.pickle','r'))



# Pepper turns
turns = {'presentation':"Hi nice to meet you I'm Pepper, what's your name?",
		 'explanation_1':"It's easy, just put the cube on a pale blue cell and tell me when you are ready.",
		 'error':"It seems there is something wrong here... let's try again",
		 'allright':"Wow, you got them all! Now let's see if I can do it...",
		 'final_greet':"Ok "+nome.split('_')[0]+" we're done for today, let's continue tomorrow."}



#--------------------PLAN--(explicit FSM)----------------------
#--------------------------------------------------------------

begin()

pepper = sar_cmd.robot


# start the interaction, awareness and face detection/recognition
voc = ['hey Pepper', 'Hi Pepper']
answer = pepper.ask('', voc)

pepper.startBasicAwareness()
pepper.startFaceDetection()

while pepper.got_face == False: time.sleep(0.5)

if patient_data['personali']['nome'] == 'unk':
	
	voc = [nome.split('_')[0]]
	answer = pepper.ask(turns['presentation'], voc)

	patient_data['personali']['nome'] = answer
	patient_data['personali']['durata'] = durata
	
else:
	voc = ['fine','ok','not so good']
	answer = pepper.ask("Hi "+patient_data['personali']['nome']+" how are you today?", voc)

	if answer == 'fine' or answer == 'ok':
		pepper.asay2("Very good, nice to hear it")
	else:
		pepper.asay2("Mmmm, come on you're going to feel better today don't worry...")



voc = ['yes','no']
answer = pepper.ask(patient_data['personali']['nome']+" are you ready to begin the session number "+str(patient_data['personali']['sessione']+1)+"?",voc)


# start the session
if answer=='yes':
	patient_data['personali']['seduta']+=1

	pepper.asay2("Ok, let's get started")
	pepper.asay2('To start drawing just put the box on a pale blue cell and tell me when you are ready')
	pepper.asay2('When you think you are done with the drawings call me again')
	
	r=0
	while r<ripetizioni:
		
		all_right = False
		while not all_right:
			drawn_moveslbl, pepper_moveslbl, session_data, ttc = [],[],[],[]
			exercise.reset_imgs()
			sampled_moveslbl = exercise.pepper_img()
			pepper.show(server)

			# this is the exercise itself 
			# (in each session the patient draws 4 different moves)
			for i in range(4):
				
				pepper.asay2('Draw move number '+str(i+1))

				voc = ['ready','start']
				answer = pepper.ask('', voc)

				
				# OpenCV trajectory tracking-----------------------------------------
				start_time = time.time()
				print('Start capture')
				cap = cv2.VideoCapture(device + cv2.CAP_V4L)
				cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280) 
				cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
				curr_trajectory = exercise.trajectory(cap)
				cap.release()
				print('End capture')
				end_time = time.time()

				# trajectory -> move_dict: '01':[(),(),...()]
				move_dict = exercise.discretize(curr_trajectory)
				# move_dict -> nn_movelbl: nearest_neighbor(move_dict.keys(), moves)
				nn_movelbl = exercise.nn(move_dict)
				#-------------------------------------------------------------------

				drawn_moveslbl.append(nn_movelbl)
				
				voc = ['done']
				answer = pepper.ask('', voc)

				# (tablet) render and show the nearest_neighbor of the drawn move
				img_move = exercise.render(exercise.moves[nn_movelbl])
				cv2.imwrite('./tablet/'+str(i+1)+'_c.png', img_move)
				pepper.show(server)

				# data collection
				t = end_time-start_time
				ttc.append(t)
				session_data.append([patients_data['personali']['seduta'],
									r,sampled_moveslbl[i],nn_movelbl,t,trajectory])


			i=0
			for sampled,drawn in zip(sampled_moveslbl,drawn_moveslbl):
				if sampled == drawn: i+=1


			if i != 4: 
				pepper.asay2(turns['error'])
				all_right = False
			else:
				pepper.asay2(turns['allright'])
				all_right = True


		# sampling postures according to p_error(move)
		for drawn in drawn_moveslbl:
			pepper_moveslbl.append(int_to_moveslbl[numpy.argmax(numpy.random.multinomial(1,
				patient_data['perrore'][drawn],
				size=1))])
		


		# start Pepper movement
		for p,drawn in zip(pepper_moveslbl,drawn_moveslbl):
			
			pepper.stopBasicAwareness()
			
			pepper.setPosture(posture.posture_dict[p], sec=mean(ttc)*0.4)
			voc = ['yes','no']
			answer = pepper.ask("Is it my "+drawn+"?", voc)

			pepper.startBasicAwareness()

			if answer == 'yes':
				pepper.asay2('Great!')

				for index in range(len(patient_data['perrore'][drawn])):
					if int_to_moveslbl[index] == drawn:
						patient_data['perrore'][drawn][index]+=massa
					else:
						patient_data['perrore'][drawn][index]-=(massa/9.0)
			else:
				pepper.asay2('Oh no, what is it then?')

				voc = [i for i in patient_data['perrore'].keys()]
				answer = pepper.ask('', voc)

				pepper.asay2('Ok I will try to remember...')

				for index in range(len(patient_data['perrore'][answer])):
					if int_to_moveslbl[index] == answer:
						patient_data['perrore'][answer][index]+=massa
					else:
						patient_data['perrore'][answer][index]-=(massa/9.0)

		r+=1
			


pepper.asay2(turns['final_greet'])


# saving data
pickle.dump(patient_data,open('./'+nome+'/'+nome+'.pickle','w'))


pepper.stopFaceDetection()
pepper.stopBasicAwareness()
pepper.normalPosture()


end()