import numpy as np
import cv2
import time

import pickle
from pprint import pprint
from sklearn.neighbors import NearestNeighbors

import random
import webbrowser

n_cells = 6*6
device = 2
'''
#----------------GRID VERTICES IDENTIFICATION--------------------
#-----------------------------------------------------------------
grid_vertices = []
current = (0,0)

def draw_red_circle(event, x, y, flags, param):
	global grid_vertices,current
	if event == cv2.EVENT_LBUTTONDBLCLK: grid_vertices.append((x,y))
	if event == cv2.EVENT_MOUSEMOVE: current = (x,y)

cv2.namedWindow("img")
cv2.setMouseCallback("img", draw_red_circle, 0)

cap = cv2.VideoCapture(device,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


while True:
	ret, img = cap.read()
	img = cv2.flip(img,-1)
	cv2.putText(img,str(current), (current[0]-40,current[1]-10), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
	
	for i in grid_vertices: 
		cv2.circle(img,i,2,(255,0,0),2)
		cv2.putText(img,str(i),(i[0]-40,i[1]-10), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

	cv2.imshow("img", img)
    
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

grid_vertices.sort(key=lambda x:x[1])

base_minore = grid_vertices[2:]
base_maggiore = grid_vertices[:2]

base_maggiore.sort(key=lambda k:k[0])
base_minore.sort(key=lambda k:k[0])

grid_vertices = base_maggiore+base_minore

print(grid_vertices)
# dumping of grid_vertices
pickle.dump(grid_vertices, open('grid_vertices.pickle','wb'))






#--------------------CELL CLUSTERS IDENTIFICATION----------------------------
#----------------------------------------------------------------------------
collection = []
current = (0,0)

def draw_red_circle(event, x, y, flags, param):
	global collection,current
	if event == cv2.EVENT_LBUTTONDBLCLK: collection.append((x,y))
	if event == cv2.EVENT_MOUSEMOVE: current = (x,y)

cv2.namedWindow("img")
cv2.setMouseCallback("img", draw_red_circle, 0)

cap = cv2.VideoCapture(device,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while True:
	ret, img = cap.read()
	img = cv2.flip(img,-1)

	grid_vertices = pickle.load(open('grid_vertices.pickle','rb'))

	pts1 = np.float32([grid_vertices[0],grid_vertices[1],grid_vertices[2],grid_vertices[3]])
	pts2 = np.float32([[0,0],[640,0],[0,900],[640,900]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	img_warped = cv2.warpPerspective(img,M,(640,900))
	img_warped_copy = img_warped.copy()
	
	cv2.putText(img_warped,str(current), (current[0]-40,current[1]-10), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)	
	
	for i in collection: 
		cv2.circle(img_warped,i,2,(255,0,0),2)
		cv2.putText(img_warped,str(i),(i[0]-40,i[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
	
	cv2.imshow("img", img_warped)
    
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

cv2.imwrite('img_warped.png',img_warped_copy)


lh = int((collection[0][0]-collection[1][0])*0.8)
dc = collection[0][0]-collection[2][0]

cell_identification = []
j=0
for y in range((n_cells//6),0,-1):
	i=0
	for x in range((n_cells//6),0,-1):
		curr_centroid = (collection[0][0]-(i*dc),collection[0][1]+(j*dc))
		ul = (curr_centroid[0]-lh, curr_centroid[1]-lh)
		br = (curr_centroid[0]+lh, curr_centroid[1]+lh)
		cell_identification.append({'id':str(x-1)+str(y-1),
							'cell_centroid':curr_centroid,
							'edge_rect':[ul,br]})
		i+=1
	j+=1


pprint(cell_identification)
# dumping of cell_identification
pickle.dump(cell_identification, open('cell_identification.pickle','wb'))


img_warped = cv2.imread('img_warped.png',1)
for i in cell_identification:
	cv2.rectangle(img_warped,
		i['edge_rect'][0], i['edge_rect'][1],
		(0,0,255),1)
	cv2.putText(img_warped,i['id'],i['cell_centroid'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

cv2.imshow('img_warped',img_warped)
cv2.waitKey(0)
 '''





 #----------------------------EXERCISE-------------------------------
 #-------------------------------------------------------------------
cell_identification = pickle.load(open('cell_identification.pickle','rb'))
grid_vertices = pickle.load(open('grid_vertices.pickle','rb'))


light_red = (0,100,100)
dark_red = (10,255,255)

light_yellow = (20,100,100) 
dark_yellow = (30,255,255)

light_cyan = ( 80, 100, 100)#(80,100,100)
dark_cyan = (130, 255, 255)#(100,255,255)


light = light_cyan
dark = dark_cyan


moves = {'occhio_dx':[(4,4),(3,4),(2,4),(1,4),(1,3),(1,2),(1,1),(2,1),(3,1),(3,2),(4,2),(4,1),(4,2),(4,3),(4,4)],
		 'occhio_sx':[(1,4),(2,4),(3,4),(4,4),(4,3),(4,2),(4,1),(3,1),(2,1),(2,2),(1,2),(1,1),(1,2),(1,3),(1,4)],
		 'orecchio_dx':[(4,5),(3,5),(2,5),(1,5),(1,4),(1,3),(1,2),(1,1),(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(3,2)],
		 'orecchio_sx':[(1,5),(2,5),(3,5),(4,5),(4,4),(4,3),(4,2),(4,1),(4,0),(3,0),(2,0),(1,0),(1,1),(1,2),(2,2)],
		 'testa':[(0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(5,3),(5,2),(4,2),(3,2),(3,3),(3,4),(2,4),(2,3),(2,2),(1,2),(0,2),(0,3),(0,4)],
		 'pancia':[(2,4),(3,4),(4,4),(4,3),(3,3),(2,3),(2,2),(3,2),(4,2),(4,1),(4,0),(3,0),(2,0),(2,1),(2,2)],
		 'spalla_dx':[(5,5),(4,5),(3,5),(2,5),(1,5),(0,5),(0,4),(0,3),(0,2),(0,1),(0,0)],
		 'spalla_sx':[(0,5),(1,5),(2,5),(3,5),(4,5),(5,5),(5,4),(5,3),(5,2),(5,1),(5,0)],
		 'naso':[(1,5),(1,4),(1,3),(1,2),(2,2),(3,2),(4,2),(4,3),(4,4),(4,5)],
		 'bocca':[(1,3),(2,2),(3,2),(4,3)],
		 'QM':[(1,4),(1,5),(2,5),(3,5),(4,5),(4,4),(3,3),(2,2),(3,2),(2,0),(3,0)]}



def render(move):

	if type(move) == dict: move = move.keys()

	img_h,img_w,shift,cell_dim = 300,300,5,50
	img_move = np.zeros([img_h,img_w,3],dtype=np.uint8)
	img_move[...] = [255,255,255]
	
	# (0,0) square coord
	ul,br = (0,img_h-cell_dim),(cell_dim,img_h)

	# complete the move
	# extra_cells = []
	# for i in range(len(move)-1):
	# 	a,b = move[i],move[i+1]
	# 	if abs(a[0]-b[0])>1:
	# 		for j in range(min(a[0],b[0])+1,max(a[0],b[0]),1): 
	# 			extra_cells.append((j,a[1]))
	# 	if abs(a[1]-b[1])>1:
	# 		for j in range(min(a[1],b[1])+1,max(a[1],b[1]),1):
	# 			extra_cells.append((a[0],j))

	# move += extra_cells

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
			cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0][0]
			M = cv2.moments(cnt)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			curr_trajectory.append([cx,cy])
		except:
			pass
		

		result = cv2.bitwise_and(frame, frame, mask=mask)
		
		try:
			cv2.circle(result,(cx,cy),2,(255,255,255),-1)
			cv2.polylines(result,[np.asarray(curr_trajectory)],False,(255,255,0),2)
		except:
			pass
		
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
		d = len(set(move).union(set(v)))
		ranking.append([k,n/d])

	ranking.sort(key=lambda x: x[1])

	return ranking[-1][0]

def reset_imgs():
	source_path = '.\\tablet\\exercise_file\\'
	dest_path = '.\\tablet\\'
	
	for i in range(5):
		img = cv2.imread(source_path+str(i)+'.png',1)
		cv2.imwrite(dest_path+str(i)+'_c.png', img)

def pepper_img():
	source_path = '.\\tablet\\exercise_file\\'
	dest_path = '.\\tablet\\'

	bodyparts_to_coord = {'orecchio_sx':(229,57),
						  'orecchio_dx':(102,57),
						  'occhio_sx':(191,56),
						  'occhio_dx':(142,56),
						  'testa':(166,12),
						  'naso':(167,62),
						  'bocca':(166,85),
						  'spalla_dx':(95,113),
						  'spalla_sx':(250,113),
						  'pancia':(167,238)
						  }

	bodyparts_sample = random.sample(bodyparts_to_coord.keys(),4)
	img = cv2.imread(source_path+'0.png',1)

	for index,bp in enumerate(bodyparts_sample):
		cv2.circle(img,bodyparts_to_coord[bp],10,(255,0,0),-1)
		cv2.putText(img,str(index+1), 
			(bodyparts_to_coord[bp][0]-6,bodyparts_to_coord[bp][1]+5), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

	cv2.imwrite(dest_path+'0_c.png',img)



def main():

	global moves

	reset_imgs()
	sampled_moveslbl = pepper_img()
	
	webbrowser.get().open('.\\tablet\\exercise.htm',new=0)
	
	drawn_moveslbl = []
	
	for i in range(4):
		input('Posiziona per la mossa '+str(i+1)+' e premi invio')
		
		
			
		print('Start capture')
		cap = cv2.VideoCapture(device,cv2.CAP_DSHOW)
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
		img_move = render(moves[nn_move_lbl])
		
		cv2.imwrite('.\\tablet\\'+str(i+1)+'_c.png', img_move)
		
		webbrowser.get().open('.\\tablet\\exercise.htm',new=0)

	#execute moves


	

if __name__ == "__main__":
	main()


