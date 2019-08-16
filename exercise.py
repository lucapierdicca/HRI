import numpy as np
import cv2
import time

import pickle
from pprint import pprint
from sklearn.neighbors import NearestNeighbors

import random
import webbrowser

n_cells = 6*6
device = 0


 #----------------------------EXERCISE-------------------------------
 #-------------------------------------------------------------------
cell_identification = pickle.load(open('cell_identification.pickle','rb'))
grid_vertices = pickle.load(open('grid_vertices.pickle','rb'))


light_red = (0,100,100)
dark_red = (10,255,255)

light_yellow = (20,100,100) 
dark_yellow = (30,255,255)

light_cyan = (80, 100, 100)#(80,100,100)
dark_cyan = (130, 255, 255)#(100,255,255)


light = light_cyan
dark = dark_cyan


moves = {'right eye':[(4,4),(3,4),(2,4),(1,4),(1,3),(1,2),(1,1),(2,1),(3,1),(3,2),(4,2),(4,1),(4,2),(4,3),(4,4)],
		 'left eye':[(1,4),(2,4),(3,4),(4,4),(4,3),(4,2),(4,1),(3,1),(2,1),(2,2),(1,2),(1,1),(1,2),(1,3),(1,4)],
		 'right ear':[(4,5),(3,5),(2,5),(1,5),(1,4),(1,3),(1,2),(1,1),(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(3,2)],
		 'left ear':[(1,5),(2,5),(3,5),(4,5),(4,4),(4,3),(4,2),(4,1),(4,0),(3,0),(2,0),(1,0),(1,1),(1,2),(2,2)],
		 'head':[(0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(5,3),(5,2),(4,2),(3,2),(3,3),(3,4),(2,4),(2,3),(2,2),(1,2),(0,2),(0,3),(0,4)],
		 'belly':[(2,4),(3,4),(4,4),(4,3),(3,3),(2,3),(2,2),(3,2),(4,2),(4,1),(4,0),(3,0),(2,0),(2,1),(2,2)],
		 'right shoulder':[(5,5),(4,5),(3,5),(2,5),(1,5),(0,5),(0,4),(0,3),(0,2),(0,1),(0,0)],
		 'left shoulder':[(0,5),(1,5),(2,5),(3,5),(4,5),(5,5),(5,4),(5,3),(5,2),(5,1),(5,0)],
		 'nose':[(1,5),(1,4),(1,3),(1,2),(2,2),(3,2),(4,2),(4,3),(4,4),(4,5)],
		 'mouth':[(1,3),(2,2),(3,2),(4,3)],
		 'QM':[(1,4),(1,5),(2,5),(3,5),(4,5),(4,4),(3,3),(2,2),(3,2),(2,0),(3,0)]}



def render(move):

	if type(move) == dict: move = move.keys()

	img_h,img_w,shift,cell_dim = 300,300,5,50
	img_move = np.zeros([img_h,img_w,3],dtype=np.uint8)
	img_move[...] = [255,255,255]
	
	# (0,0) square coord
	ul,br = (0,img_h-cell_dim),(cell_dim,img_h)

	for y in range(n_cells//6):
		for x in range(n_cells//6):
			if (x,y) in move: color = (255,255,0)
			else: color = (230,230,230) 
			
			cv2.rectangle(img_move,
				((ul[0]+(x*cell_dim))+shift,(ul[1]-(y*cell_dim))+shift),
				((br[0]+(x*cell_dim))-shift,(br[1]-(y*cell_dim))-shift),
				color,-1)
				

	return img_move


def frame_transform(frame):

	global grid_vertices
	
	frame = cv2.flip(frame,-1)
	#frame = frame[80:frame.shape[0]-80,:]
	pts1 = np.float32([grid_vertices[0],grid_vertices[1],grid_vertices[2],grid_vertices[3]])
	pts2 = np.float32([[0,0],[640,0],[0,900],[640,900]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	frame = cv2.warpPerspective(frame,M,(640,900))
	#frame = frame[20:300-20,20:300-20]
	frame = cv2.GaussianBlur(frame, (5,5), 3)
	frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	return frame, frame_hsv



def trajectory(cap):
	global light_blue, dark_blue, cell_identification

	curr_trajectory = []

	while True:
		ret, frame = cap.read()
		frame, frame_hsv = frame_transform(frame)
		
		mask = cv2.inRange(frame_hsv, light, dark)

		try:
			cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
			M = cv2.moments(cnt)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			curr_trajectory.append([cx,cy])
		except:
			print('contour')
		

		result = cv2.bitwise_and(frame, frame, mask=mask)
		
		try:
			cv2.circle(result,(cx,cy),2,(255,255,255),-1)
			cv2.polylines(result,[np.asarray(curr_trajectory)],False,(255,255,0),2)
		except:
			print('circle')
		
		for cell in cell_identification:
			cv2.rectangle(result,cell['edge_rect'][0],cell['edge_rect'][1],(0,0,255),2)
		
		cv2.imshow('frame',frame)
		cv2.imshow('masked',result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()
	

	return curr_trajectory




def discretize(trajectory):
	global cell_identification
	cell_centroids = [cell['cell_centroid'] for cell in cell_identification]

	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(cell_centroids)
	
	discretization = {}

	for index,cluster_id in enumerate(neigh.kneighbors(trajectory, return_distance=False)):
		if cell_identification[cluster_id[0]]['id'] not in discretization:
			discretization[(int(cell_identification[cluster_id[0]]['id'][0]),int(cell_identification[cluster_id[0]]['id'][1]))]=[trajectory[index]]
		else:
			discretization[(int(cell_identification[cluster_id[0]]['id'][0]),int(cell_identification[cluster_id[0]]['id'][1]))].append(trajectory[index])

	return discretization


def nn(move):
	global moves
	if type(move) == dict: move = list(move.keys())

	ranking = []

	for k,v in moves.items():
		
		n = len(set(move).intersection(set(v)))
		d = float(len(set(move).union(set(v))))
		ranking.append([k,n/d])

	ranking.sort(key=lambda x: x[1])

	return ranking[-1][0]


def reset_imgs():
	source_path = './tablet/exercise_file/'
	dest_path = './tablet/'
	
	for i in range(5):
		img = cv2.imread(source_path+str(i)+'.png',1)
		cv2.imwrite(dest_path+str(i)+'_c.png', img)


def pepper_img():
	source_path = './tablet/exercise_file/'
	dest_path = './tablet/'

	bodyparts_to_coord = {'left ear':(407,103),
						  'right ear':(194,103),
						  'left eye':(351,118),
						  'right eye':(250,114),
						  'head':(300,20),
						  'nose':(302,117),
						  'mouth':(301,156),
						  'right shoulder':(194,199),
						  'left shoulder':(413,205),
						  'belly':(303,425)
						  }

	bodyparts_sample = random.sample(bodyparts_to_coord.keys(),4)
	img = cv2.imread(source_path+'0.png',1)

	for index,bp in enumerate(bodyparts_sample):
		cv2.circle(img,bodyparts_to_coord[bp],10,(255,0,0),-1)
		cv2.putText(img,str(index+1), 
			(bodyparts_to_coord[bp][0]-6,bodyparts_to_coord[bp][1]+5), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

	cv2.imwrite(dest_path+'0_c.png',img)

	return bodyparts_sample



def main():

	global moves

	reset_imgs()
	sampled_moveslbl = pepper_img()
	
	webbrowser.get().open('./tablet/index.htm',new=0)
	
	drawn_moveslbl = []
	
	for i in range(4):
		raw_input('Posiziona per la mossa '+str(i+1)+' e premi invio')
		
		
			
		print('Start capture')
		cap = cv2.VideoCapture(device + cv2.CAP_V4L)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280) 
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
		curr_trajectory = trajectory(cap)
		cap.release()
		print('End capture')

		# trajectory -> move_dict: '01':[(),(),...()]
		move_dict = discretize(curr_trajectory)
		# move_dict -> nn_movelbl: nearest_neighbor(move_dict.keys(), moves)
		nn_movelbl = nn(move_dict)
		drawn_moveslbl.append(nn_movelbl)
		# draw the nearest_neighbor
		img_move = render(moves[nn_movelbl])
		
		cv2.imwrite('./tablet/'+str(i+1)+'_c.png', img_move)
		
		webbrowser.get().open('./tablet/index.htm',new=0)

	#execute moves


	

if __name__ == "__main__":
	main()



