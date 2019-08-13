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

    
def oldPatient():

	global pepper

	pepper.asay2("Oh hi {}, welcome back. I'm Pepper. Do you want to play?".format(pepper.faceLabel) )

	answer = pepper.asr(["yes", "no"])
	if answer == "yes":
	    pepper.asay2("This is your session number " + str(len(patients_info()[1][pepper.patient]) + 1) + "The game is about to start. Get Ready")
	    print("Starting Game")
	    #self.stopFaceDetection()
	    #end()
	elif answer == "no":
	    print("Pepper Sad")
	    pepper.say("PEPPER SAD")
	    pepper.stopFaceDetection()

	    end()




# from patients patients_info[0] = list of patients names, patients_info[1] = dictionary of names and lists of scores

def newPatient():
	global pepper

	pepper.asay2("Hi buddy. What's your name?")

	#pepper.face_service = pepper.session.service("ALFaceDetection")
	voc = patients_info()[0] + ["stop"]
	answer = pepper.asr(vocabulary = voc)
	pepper.face_service.learnFace(answer)
	pepper.patient = pepper.faceLabel
	if answer == "stop":
	    pepper.stopFaceDetection()

	    end()
	else:

	    pepper.face_service.reLearnFace(answer)

	    pepper.asay2("Nice to meet you {}. I'll remember you for next sessions.".format(answer))

	    print pepper.face_service.getLearnedFacesList()

	    # time.sleep(1)
	    # pepper.face_service.reLearnFace(asr_word)
	    # pepper.say("Now turn your face left")
	    # time.sleep(1)
	    # pepper.face_service.reLearnFace(asr_word)
	    # pepper.say("Now turn your face right")
	    # time.sleep(1)
	    # pepper.face_service.reLearnFace(asr_word)

	    pepper.asay2("Ok, this is your first sesssion. Let's begin. STARTING GAME")

	    print("INIZIO GIOCO")
	    pepper.stopFaceDetection()

	    end()


patient_data = {'personali':{'nome':'unk',
								  'durata':0,
								  'sessione':0},
				'esercizi':[],
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
ripetizioni = 5
device = 0

print(os.listdir('./'+nome))

if len(os.listdir('./'+nome)) == 0:
	pickle.dump(patient_data,open('./'+nome+'/'+nome+'.pickle','w'))
else:
	patient_data = pickle.load(open('./'+nome+'/'+nome+'.pickle','r'))



err=0




turns = {'presentation':"Listen "+nome.split('_')[0]+" I'm a bit confused these days. ...",
		 'explanation_1':"It's easy, just put the cube on a pale blue cell and tell me when you are ready.",
		 'error':"It seems there is something wrong here... let's try again",
		 'all_right':"Very good, you got them all. Now let's see what I can do...",
		 'final_greet':"Ok "+nome.split('_')[0]+" we're done for today, let's continue tomorrow."}


answer = ''

#---------PLAN------------

begin()

pepper = sar_cmd.robot

try:
	
	voc = ['Hey Pepper', 'Hi Pepper']
	while answer not in voc:
		answer = pepper.asr(vocabulary=voc)
		answer = answer[answer.find('> ')+2:answer.find(' <')] 
		time.sleep(0.5)


	pepper.startBasicAwareness()
	pepper.startFaceDetection_()

	while pepper.check == False: time.sleep(0.5)

	if pepper.old_patient:
		oldPatient()
	else:
		newPatient()
	
	patient_data['personali']['nome'] = pepper.patient
	patient_data['personali']['durata'] = durata
	
	#while pepper.got_face == False: time.sleep(0.5)


	# if patient_data['personali']['nome'] == 'unk':
	# 	pepper.asay2("Hi nice to meet you I'm Pepper, what's your name?")
		
	# 	voc = [nome.split('_')[0]]
	# 	while answer not in voc:
	# 		answer = pepper.asr(vocabulary=voc)
	# 		answer = answer[answer.find('> ')+2:answer.find(' <')] 
	# 		time.sleep(0.5)

	# 	patient_data['personali']['nome'] = answer
	# 	patient_data['personali']['durata'] = durata
		
	# else:
	# 	pepper.asay2("Hi "+patient_data['personali']['nome']+" how are you today?")

	# 	voc = ['fine','ok','not so good']
	# 	while answer not in voc:
	# 		answer = pepper.asr(vocabulary=voc)
	# 		answer = answer[answer.find('> ')+2:answer.find(' <')] 
	# 		time.sleep(0.5)

	# 	if answer == 'fine' or answer == 'ok':
	# 		pepper.asay2("Very good, nice to hear it")
	# 	else:
	# 		pepper.asay2("Mmmm, come on you're going to feel better today...")

	patient_data['personali']['sessione']+=len(patients_info()[1][pepper.patient]) + 1


	pepper.asay2(patient_data['personali']['nome']+" are you ready to begin the session number "+str(patient_data['personali']['sessione'])+"?")
	voc = ['yes','no']
	while answer not in voc: 
		answer = pepper.asr(vocabulary=voc)
		answer = answer[answer.find('> ')+2:answer.find(' <')] 
		time.sleep(0.5)

	if answer=='yes':
		pepper.asay2("Ok, let's get started")
		pepper.asay2('To start drawing just put the box on a pale blue cell and tell me when you are ready')
		pepper.asay2('When you think you are done with the dranwings call me again')
		
		all_right = False
		while not all_right:
			drawn_moveslbl = []
			pepper_moveslbl = []
			ttc = []
			exercise.reset_imgs()
			pepper.show(server)
			time.sleep(1)
			sampled_moveslbl = exercise.pepper_img()
			print('SAMPLED', sampled_moveslbl)
			pepper.show(server)

			
			for i in range(4):

				pepper.say('Draw move number '+str(i+1))

				voc = ['ready','start']
				while answer not in voc: 
					answer = pepper.asr(vocabulary=voc)
					answer = answer[answer.find('> ')+2:answer.find(' <')]  
					time.sleep(0.5)

				start_time = time.time()
								
				print('Start capture')
				cap = cv2.VideoCapture(device + cv2.CAP_V4L)
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
				

				voc = ['done']
				while answer not in voc: 
					answer = pepper.asr(vocabulary=voc)
					answer = answer[answer.find('> ')+2:answer.find(' <')] 
					time.sleep(0.5)

				end_time = time.time()

				ttc.append(end_time-start_time)

				print(nn_movelbl)

				# draw the nearest_neighbor
				img_move = exercise.render(exercise.moves[nn_movelbl])
				cv2.imwrite('./tablet/'+str(i+1)+'_c.png', img_move)
				pepper.show(server)


			i=0
			for sampled,drawn in zip(sampled_moveslbl,drawn_moveslbl):
				if sampled == drawn: i+=1


			if i != 4: 
				pepper.asay2(turns['error'])
				all_right = False
			else:
				pepper.asay2("Wow, you got them all!")
				pepper.asay2("Now let's see if I can do it...")
				all_right = True
		
		
		print(drawn_moveslbl)

		print(mean(ttc))



		# SAMPLING CON P_ERRORE
		for i in drawn_moveslbl:
			pepper_moveslbl.append(int_to_moveslbl[numpy.argmax(numpy.random.multinomial(1,
				patient_data['perrore'][i],
				size=1))])
		


		# MOVIMENTO PEPPER
		for p,d in zip(pepper_moveslbl,drawn_moveslbl):
			pepper.stopBasicAwareness()
			pepper.setPosture(posture.posture_dict[p], sec=mean(ttc)*0.4)
			print('PEPPER MOVE ',p)
			pepper.asay2("Is it my "+d+"?")

			voc = ['yes','no']
			while answer not in voc: 
				answer = pepper.asr(vocabulary=voc)
				answer = answer[answer.find('> ')+2:answer.find(' <')]
				time.sleep(0.5)

			pepper.startBasicAwareness()

			if answer == 'yes':
				pepper.asay2('Great!')
				for index in range(len(patient_data['perrore'][d])):
					if int_to_moveslbl[index] == d:
						patient_data['perrore'][d][index]+=massa
					else:
						patient_data['perrore'][d][index]-=(massa/9.0)
			else:
				pepper.asay2('Oh no, what is it then?')

				voc = [i for i in patient_data['perrore'].keys()]
				while answer not in voc: 
					answer = pepper.asr(vocabulary=voc)
					answer = answer[answer.find('> ')+2:answer.find(' <')] 
					time.sleep(0.5)

				pepper.asay2('Ok I will try to remember...')

				for index in range(len(patient_data['perrore'][answer])):
					if int_to_moveslbl[index] == answer:
						patient_data['perrore'][answer][index]+=massa
					else:
						patient_data['perrore'][answer][index]-=(massa/9.0)

			answer = ''

	pprint(patient_data['perrore'])
	pepper.asay2(turns['final_greet'])
	

	# saving data
	pickle.dump(patient_data,open('./'+nome+'/'+nome+'.pickle','w'))

except RuntimeError:
	err = 1
	pepper.stopFaceDetection_()
	pepper.stopBasicAwareness()
	pepper.normalPosture()

if err==0:
	pepper.stopFaceDetection_()
	pepper.stopBasicAwareness()
	pepper.normalPosture()



end()