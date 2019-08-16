import os, sys
import exercise
import pickle
from pprint import pprint
import cv2
import numpy
import time
import webbrowser



def asr(voc):
	transcription = ''
	while transcription not in voc:
		transcription = raw_input('Scrivi\n')

	return transcription


def say(phrase):
	print(phrase)


def show():
	webbrowser.get().open('./tablet/index.htm',new=0)



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


nome = 'Luca_Pierdicca'
durata = 5
massa = 0.1
ripetizioni = 5

device = 0



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


answer = ''

#---------PLAN------------




voc = ['hey pepper', 'hi pepper']
answer = asr(voc)

if patient_data['personali']['nome'] == 'unk':
	say(turns['presentation'])
	
	voc = [nome.split('_')[0].lower()]
	answer = asr(voc)


	patient_data['personali']['nome'] = answer
	patient_data['personali']['durata'] = durata

else:
	say("Hi "+patient_data['personali']['nome']+" how are you today?")

	voc = ['fine','ok','not so good']
	answer = asr(voc)

	if answer == 'fine' or answer == 'ok':
		say("Very good, nice to hear it")
	else:
		say("Mmmm, come on you're going to feel better today don't worry...")


patient_data['personali']['sessione']+=1

say(patient_data['personali']['nome']+" are you ready to begin the session number "+str(patient_data['personali']['sessione'])+"?")

voc = ['yes','no']
answer = asr(voc)


if answer=='yes':
	say("Ok, let's get started")
	say('To start drawing just put the box on a pale blue cell and tell me when you are ready')
	say('When you think you are done with the drawing call me again')

	r=0
	while r<ripetizioni:
		all_right = False
		while not all_right:
			drawn_moveslbl, pepper_moveslbl, session_data, ttc = [],[],[],[]
			exercise.reset_imgs()
			sampled_moveslbl = exercise.pepper_img()
			show()

			# this is the exercise itself 
			# (in each session the patient draws 4 different moves)		
			for i in range(4):

				say('Draw move number '+str(i+1))

				voc = ['ready','start']
				answer = asr(voc)

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
				answer = asr(voc)
				
				img_move = exercise.render(exercise.moves[nn_movelbl])
				cv2.imwrite('./tablet/'+str(i+1)+'_c.png', img_move)
				show()

				# data collection
				t = end_time-start_time
				ttc.append(t)
				session_data.append([patients_data['personali']['seduta'],
									r,sampled_moveslbl[i],nn_movelbl,t,trajectory])


			i=0
			for sampled,drawn in zip(sampled_moveslbl,drawn_moveslbl):
				if sampled == drawn: i+=1


			if i != 4: 
				say(turns['error'])
				all_right = False
			else:
				say(turns['allright'])
				all_right = True
		



		# sampling postures according to p_error(move)
		for i in drawn_moveslbl:
			pepper_moveslbl.append(int_to_moveslbl[numpy.argmax(numpy.random.multinomial(1,
				patient_data['perrore'][i],
				size=1))])
		


		# MOVIMENTO PEPPER
		for p,d in zip(pepper_moveslbl,drawn_moveslbl):
			print('PEPPER POSTURE',p)
			say("Is it my "+d+"?")

			voc = ['yes','no']
			answer = asr(voc)

			if answer == 'yes':
				say('Great!')
				for index in range(len(patient_data['perrore'][d])):
					if int_to_moveslbl[index] == d:
						patient_data['perrore'][d][index]+=massa
					else:
						patient_data['perrore'][d][index]-=(massa/9.0)
			else:
				say('Oh no, what is it then?')

				voc = [i for i in patient_data['perrore'].keys()]
				answer = asr(voc)

				say('Ok I will try to remember...')

				for index in range(len(patient_data['perrore'][answer])):
					if int_to_moveslbl[index] == answer:
						patient_data['perrore'][answer][index]+=massa
					else:
						patient_data['perrore'][answer][index]-=(massa/9.0)

		r+=1
		answer = ''

pprint(patient_data['perrore'])
say(turns['final_greet'])


# saving data
pickle.dump(patient_data,open('./'+nome+'/'+nome+'.pickle','w'))




