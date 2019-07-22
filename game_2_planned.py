import numpy as np
import cv2
import time

import pickle
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import threading

n_cells = 6*6


#----------------GRID VERTICES IDENTIFICATION--------------------
#-----------------------------------------------------------------

'''
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOCUS, 200) 
time.sleep(2)
ret, frame = cap.read()
cap.release()

frame = cv2.flip(frame,-1)
frame = frame[80:frame.shape[0]-80,:]
frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



#frame = cv2.medianBlur(frame,5)
#frame_bw = cv2.threshold(frame_bw,185,255,cv2.THRESH_BINARY)[1]
#frame_bw = cv2.adaptiveThreshold(frame_bw,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
#kernel = np.ones((5,5),np.uint8)
#frame_bw = cv2.erode(frame_bw,kernel,iterations = 1)
#kernel = np.ones((3,3),np.uint8)
#frame_bw = cv2.dilate(frame_bw,kernel,iterations = 2)

corners = cv2.goodFeaturesToTrack(frame_bw,4,0.1,15)
corners = np.int0(corners)

grid_vertices = []

for i in corners:
    x,y = i.ravel()
    grid_vertices.append((x,y))
    cv2.circle(frame,(x,y),3,(255,0,0),-1)

grid_vertices.sort(key=lambda k:k[1])

base_minore = grid_vertices[2:]
base_maggiore = grid_vertices[:2]

base_maggiore.sort(key=lambda k:k[0])
base_minore.sort(key=lambda k:k[0])

grid_vertices = base_maggiore+base_minore


print(grid_vertices)
# dumping of grid_vertices
pickle.dump(grid_vertices, open('grid_vertices.pickle','wb'))

cv2.imwrite('frame.png',frame)
cv2.imshow('image',frame)
cv2.waitKey(0)

'''




#--------------------CELL CLUSTERS IDENTIFICATION----------------------------
#----------------------------------------------------------------------------
'''
THRESHOLD = 190
ITERATIONS = 2

cap = cv2.VideoCapture(0)
time.sleep(2)
ret, frame = cap.read()
cap.release()


frame = cv2.flip(frame,-1)
frame = frame[80:frame.shape[0]-80,:]
grid_vertices = pickle.load(open('grid_vertices.pickle','rb'))

pts1 = np.float32([grid_vertices[0],grid_vertices[1],grid_vertices[2],grid_vertices[3]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
frame_warped = cv2.warpPerspective(frame,M,(300,300))

cv2.imshow('image',frame_warped)
cv2.waitKey(0)

frame_warped_bw = cv2.cvtColor(frame_warped, cv2.COLOR_BGR2GRAY)
frame_warped_th = cv2.threshold(frame_warped_bw,THRESHOLD,255,cv2.THRESH_BINARY_INV)[1]
frame_warped_th = frame_warped_th[20:300-20,20:300-20]
frame_warped = frame_warped[20:300-20,20:300-20]
kernel = np.ones((3,3),np.uint8)
frame_warped_th = cv2.erode(frame_warped_th,kernel,iterations = ITERATIONS)
cv2.imshow('frame_warped',frame_warped_th)
cv2.waitKey(0)

# k-means
points_coord = np.argwhere(frame_warped_th==255)

kmeans = KMeans(n_clusters=n_cells, random_state=0).fit(points_coord)

for i in kmeans.cluster_centers_:
	cv2.circle(frame_warped,(int(i[0]),int(i[1])), 6, (0,0,0), -1)

cv2.imshow('frame_warped',frame_warped)
cv2.imwrite('cell_centroid.png',frame_warped)
cv2.waitKey(0)

# cell coordinate assigment
cluster_centers_coord = [[coord[0],coord[1]] for coord in kmeans.cluster_centers_]
cluster_centers_coord.sort(key= lambda c: c[0])
column_subdivision = []
for i in range(0,n_cells, n_cells//2):
	column_subdivision.append(cluster_centers_coord[i:i+n_cells//2])

for i in column_subdivision:
	i.sort(key= lambda c: c[1], reverse=True)

cluster_centers_coord = []
for i in column_subdivision:
	for j in i:
		cluster_centers_coord.append(j)

cx,cy = 0,n_cells//2
for index,i in enumerate(cluster_centers_coord):
	i+=[cx,int(index%cy)]
	if(index%cy == 1): cx+=1


for coord in cluster_centers_coord:
	for index,orderedcoord in enumerate(kmeans.cluster_centers_):
		if coord[0] == orderedcoord[0] and coord[1] == orderedcoord[1]:
			coord.append(index)

for coord in cluster_centers_coord:
	temp = [point for point,label in zip(points_coord,kmeans.labels_) if label == coord[4]]
	temp = np.asarray(temp)
	upper_left_coord = (np.min(temp[:,0]),np.min(temp[:,1])) 
	bottom_right_coord = (np.max(temp[:,0]),np.max(temp[:,1]))
	coord+=[upper_left_coord,bottom_right_coord]


for coord in cluster_centers_coord:
	cv2.rectangle(frame_warped,coord[5],coord[6],(0,0,255),1)



cell_identification = []

for coord in cluster_centers_coord:
	cell_identification.append({'id':str(coord[2])+str(coord[3]),
								'cell_centroid':(int(coord[0]),int(coord[1])),
								'cluster_lbl':coord[4],
								'edge_rect':[coord[5],coord[6]]})

# dumping of cell_identification
pickle.dump(cell_identification, open('cell_identification.pickle','wb'))

pprint(cell_identification)

cv2.imshow('frame_warped',frame_warped)
cv2.imwrite('rectangle.png',frame_warped)
cv2.waitKey(0)
'''
 





 #----------------GLOBAL_VAR---------------
cell_identification = pickle.load(open('cell_identification.pickle','rb'))

grid_vertices = pickle.load(open('grid_vertices.pickle','rb'))



X = np.array([[1,0],
			  [0,1],
			  [-1,0],
			  [0,-1]])
y = ['E','N','O','S']

light_red = (0,100,100)
dark_red = (10,255,255)

light_yellow = (20,100,100) 
dark_yellow = (30,255,255)

light_blue = (100,150,0)
dark_blue = (140,255,255)


#color = np.uint8([[[255,255,0]]])
#hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)

light = (80,100,100)
dark = (100,255,255)


state = np.zeros((2,2,2))

id_to_color = {1:'red',2:'yellow'}
id_to_bgr = {1:(0,0,255),2:(0,255,255)}
color_to_id = {v:k for k,v in id_to_color.items()}
id_to_orie = {1:'N',2:'E',3:'S',4:'O'}
orie_to_id = {v:k for k,v in id_to_orie.items()}




moves = {'OcDX':[(4,4),(1,4),(1,1),(3,1),(3,2),(4,2),(4,1),(4,4)],
		 'OcSX':[(1,4),(4,4),(4,1),(2,1),(2,2),(1,2),(1,1),(1,4)],
		 'OrDX':[(4,5),(1,5),(1,0),(4,0),(4,2),(3,2)],
		 'OrSX':[(1,5),(4,5),(4,0),(1,0),(1,2),(2,2)],
		 'T':[(0,4),(5,4),(5,2),(3,2),(3,4),(2,4),(2,2),(0,2),(0,4)],
		 'P':[(2,4),(4,4),(4,3),(2,3),(2,2),(4,2),(4,0),(2,0),(2,2)],
		 'SpDX':[(5,5),(0,5),(0,0)],
		 'SpSX':[(0,5),(5,5),(5,0)],
		 'N':[(1,5),(1,2),(4,2),(4,5)]}



def drawMove(move):
	img_h,img_w,shift,cell_dim = 300,300,5,50
	img_grid = np.zeros([img_h,img_w,3],dtype=np.uint8)
	img_grid[...] = [255,255,255]
	
	ul,br = (0,img_h-cell_dim),(cell_dim,img_h)

	for y in range(n_cells//6):
		for x in range(n_cells//6):
			cv2.rectangle(img_grid,
				((ul[0]+(x*50))+shift,(ul[1]-(y*50))+shift),
				((br[0]+(x*50))-shift,(br[1]-(y*50))-shift),
				(185,185,185),-1)
				

	cv2.imshow('img',img_grid)
	cv2.waitKey(0)








def pca(mask,ul,br):
	points_coord = np.argwhere(mask==255)

	x_prime = points_coord[:,1]
	y_prime = (br[1]-ul[1])-points_coord[:,0]
	
	x_prime_mean = round(np.mean(x_prime))
	y_prime_mean = round(np.mean(y_prime))

	x_prime = x_prime - x_prime_mean
	y_prime = y_prime - y_prime_mean

	points_coord_prime = np.vstack((x_prime,y_prime))
	points_coord_prime = points_coord_prime.T
	
	pca = PCA(n_components=1, svd_solver='full')
	pca.fit(points_coord_prime) 
	x_pca = pca.components_[0][0]
	y_pca = pca.components_[0][1]

	compo = np.array([x_pca,y_pca])
	
	return compo

def frame_transform(frame):

	global grid_vertices

	#b,g,r = cv2.split(frame)
	#r-=40; g-=20;
	#frame = cv2.merge((b,g,r))
	
	frame = cv2.flip(frame,-1)
	frame = frame[80:frame.shape[0]-80,:]
	pts1 = np.float32([grid_vertices[0],grid_vertices[1],grid_vertices[2],grid_vertices[3]])
	pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	frame = cv2.warpPerspective(frame,M,(300,300))
	frame = frame[20:300-20,20:300-20]
	frame = cv2.GaussianBlur(frame, (5,5), 3)
	frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	return frame, frame_hsv


def sense(frame_hsv):

	global state,cell_identification,light_red,dark_red,light_yellow,dark_yellow

	for cell in cell_identification:
		ul=cell['edge_rect'][0]
		br=cell['edge_rect'][1]
		cell_area = frame_hsv[ul[1]:br[1], ul[0]:br[0]]
		mask_red = cv2.inRange(cell_area, light_red, dark_red)
		mask_yellow = cv2.inRange(cell_area, light_yellow, dark_yellow)
		
		if np.count_nonzero(mask_red) > 500:
			
			compo = pca(mask_red,ul,br) 
			nn = np.argmin(np.sum(np.square(X-compo),axis=1))
			
			state[int(cell['id'][0]),int(cell['id'][1]),0] = color_to_id['red']
			state[int(cell['id'][0]),int(cell['id'][1]),1] = orie_to_id[y[nn]]

		if np.count_nonzero(mask_yellow) > 500:

			compo = pca(mask_yellow,ul,br)
			nn = np.argmin(np.sum(np.square(X-compo),axis=1))

			state[int(cell['id'][0]),int(cell['id'][1]),0] = color_to_id['yellow']
			state[int(cell['id'][0]),int(cell['id'][1]),1] = orie_to_id[y[nn]]

		if not(np.count_nonzero(mask_red) > 500) and not(np.count_nonzero(mask_yellow) > 500):
			
			state[int(cell['id'][0]),int(cell['id'][1]),0] = 0
			state[int(cell['id'][0]),int(cell['id'][1]),1] = 0


def display(generic_state):

	global cell_identification,id_to_bgr
	grid_img = cv2.imread('cell_centroid.png',1)
	
	for i in range(n_cells//2):
		for j in range(n_cells//2):
			if (generic_state[i,j,:] != [0,0]).any():
				for cell in cell_identification:
					if int(cell['id'][0]) == i and int(cell['id'][1]) == j:

						ul=cell['edge_rect'][0]
						br=cell['edge_rect'][1]

						color_gbr = id_to_bgr[generic_state[i,j,0]]

						# NORD
						tip = np.array([ul[0]+(br[0]-ul[0])//2,ul[1]+((br[1]-ul[1])//2)-30])
						tail = np.array([ul[0]+(br[0]-ul[0])//2,ul[1]+((br[1]-ul[1])//2)+30])

						mean = np.mean(np.vstack((tip,tail)), axis=0)

						T_origin = np.vstack((tip,tail))-mean

						tip = np.reshape(T_origin[0,:],(2,1))
						tail = np.reshape(T_origin[1,:],(2,1))

						theta = 0


						if id_to_orie[generic_state[i,j,1]] == 'O': theta = 90
						if id_to_orie[generic_state[i,j,1]] == 'S': theta = 180
						if id_to_orie[generic_state[i,j,1]] == 'E': theta = 270


						R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
									  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
						

						tip = np.reshape(R.T@tip,(-1,))
						tail = np.reshape(R.T@tail,(-1,))

						tip+=mean
						tail+=mean

						cv2.arrowedLine(grid_img,tuple(tail.astype(int)),tuple(tip.astype(int)),color_gbr,3,tipLength=0.3)

	cv2.imshow('img',grid_img)
	cv2.waitKey(0)

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
	

	return curr_trajectory


def discretize(trajectory):
	global cell_identification
	cell_centroids = [cell['cluster_centroid'] for cell in cell_identification]

	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(cell_centroids)
	

	discretization = {}

	for index,cluster_id in enumerate(neigh.kneighbors(trajectory, return_distance=False)):
		if cell_identification[cluster_id[0]]['id'] not in discretization:
			discretization[cell_identification[cluster_id[0]]['id']]=[trajectory[index]]
		else:
			discretization[cell_identification[cluster_id[0]]['id']].append(trajectory[index])

	return discretization

def render(discretization):
	global cell_identification

	cellid_to_cellrect = {cell['id']:cell['edge_rect'] for cell in cell_identification}
	img = np.zeros([260,260,3],dtype=np.uint8)
	img[...] = [255,255,255]

	for cellid in discretization.keys():
		ul = cellid_to_cellrect[cellid][0]
		br = cellid_to_cellrect[cellid][1]
		cv2.rectangle(img,ul,br,(255,255,0),-1)

	return img

def start_thread(target,cap):
	x = threading.Thread(target=target, args=(cap,))
	x.start()


def to_PDDL(init_goal_state, problem_file):
	global id_to_color, id_to_orie
	init_goal_PDDL = []

	problem_file = open(problem_file,'r').read()
	print(problem_file)
	
	for index,s in enumerate(init_goal_state):
		temp = []
		for i in range(n_cells//2):
			for j in range(n_cells//2):
				# (x,y,color,orientation)
				current_predicate_tuple = (i,j,s[i,j,0],s[i,j,1])
				# index == 0 -> (empty  ) solo per init 
				if current_predicate_tuple[2] == 0 and current_predicate_tuple[3] == 0 and index==0:
					temp.append("(empty p%d%d)" % (i,j))
				else:
					temp.append("(at %s p%d%d)(orie %s %s)" % (id_to_color[current_predicate_tuple[2]],i,j,
						id_to_color[current_predicate_tuple[2]],id_to_orie[current_predicate_tuple[3]]))
		init_goal_PDDL.append("".join(temp))
	pprint(init_goal_PDDL)






			
def main():
	

	while True:
		
		print('Start capture')
		cap = cv2.VideoCapture(0)
		curr_trajectory = trajectory(cap)
		cap.release()
		print('End capture')

		discretization = discretize(curr_trajectory)

		move = render(discretization)

		cv2.imshow('move',move)
		
		print('Waiting')
		if cv2.waitKey(3000) & 0xFF == ord('q'):
			break

	


drawMove(0)






















'''

cap = cv2.VideoCapture(0)

while(True):

	ret, frame = cap.read()
	frame, frame_hsv = frame_transform(frame)
	
	mask_red = cv2.inRange(frame_hsv, light_red, dark_red)
	mask_yellow = cv2.inRange(frame_hsv, light_yellow, dark_yellow)
	all_mask = mask_red+mask_yellow
	result = cv2.bitwise_and(frame, frame, mask=all_mask)
	


	for cell in cell_identification:
		ul=cell['edge_rect'][0]
		br=cell['edge_rect'][1]
		cell_area = frame_hsv[ul[1]:br[1], ul[0]:br[0]]
		mask_red = cv2.inRange(cell_area, light_red, dark_red)
		mask_yellow = cv2.inRange(cell_area, light_yellow, dark_yellow)

		if np.count_nonzero(mask_yellow) > 500:

			compo = pca(mask_yellow)
			nn = np.argmin(np.sum(np.square(X-compo),axis=1))

			print(cell['id']+'__'+'yellow'+'__'+y[nn])
		
		if np.count_nonzero(mask_red) > 500:
			
			compo = pca(mask_red) 
			nn = np.argmin(np.sum(np.square(X-compo),axis=1))
			
			print(cell['id']+'__'+'red'+'__'+y[nn])

	print()

	time.sleep(0.5)
	cv2.imshow('masked',result)
	cv2.imshow('frame',frame)
	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()

'''























'''
init_state = [0,0,0,0,0]
goal_state = [1,1,1,1,1]
curr_state = init_state



while True:
	if goal_state == curr_state:
		break
	else:
		x,y = observe()
		if (curr_state[0] != x) or (curr_state[1] != y):
			
			curr_state[0] = x
			curr_state[1] = y
			
			if curr_state == [3,4,0,1,1]:
				say('asdasda')
			elif curr_state == [5,4,1,1,1]:
				say('asdasda?')
				ans = asr()
				if ans == 'yes':
					grab()
'''




































# select hsv manually
'''
import cv2
import numpy as np


cap = cv2.VideoCapture(0)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'result',0,179,nothing)
cv2.createTrackbar('s', 'result',0,255,nothing)
cv2.createTrackbar('v', 'result',0,255,nothing)

while(1):

    _, frame = cap.read()

    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # get info from track bar and appy to result
    h = cv2.getTrackbarPos('h','result')
    s = cv2.getTrackbarPos('s','result')
    v = cv2.getTrackbarPos('v','result')

    # Normal masking algorithm
    lower_blue = np.array([h,s,v])
    upper_blue = np.array([180,255,255])

    mask = cv2.inRange(hsv,lower_blue, upper_blue)

    result = cv2.bitwise_and(frame,frame,mask = mask)

    cv2.imshow('result',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
'''


# contours
'''
mask_red = cv2.inRange(frame_hsv, light_red, dark_red)
mask_yellow = cv2.inRange(frame_hsv, light_yellow, dark_yellow)

cnt_orange = cv2.findContours(mask_red, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0][0]
cnt_black = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0][0]


M_orange = cv2.moments(cnt_orange)
try:
	cx_orange = int(M_orange['m10']/M_orange['m00'])
	cy_orange = int(M_orange['m01']/M_orange['m00'])
except:
	print('div_zero')

M_black = cv2.moments(cnt_black)
try:
	cx_black = int(M_black['m10']/M_black['m00'])
	cy_black = int(M_black['m01']/M_black['m00'])
except:
	print('div_zero')


all_mask = mask_red+mask_yellow
result = cv2.bitwise_and(frame, frame, mask=all_mask)
all_mask = np.reshape(all_mask,(all_mask.shape[0],all_mask.shape[1],-1))

e = np.dstack((all_mask,all_mask))
e = np.dstack((e,all_mask))

result[np.where((e==[0,0,0]).all(axis=2))] = [255,255,255];


cv2.circle(result,(cx_orange,cy_orange),3,(255,255,255),-1)
cv2.circle(result,(cx_black,cy_black),3,(255,255,255),-1)


'''




# # make it white (non è necessario)
# all_mask = np.reshape(all_mask,(all_mask.shape[0],all_mask.shape[1],-1))
# e = np.dstack((all_mask,all_mask))
# e = np.dstack((e,all_mask))
# result[np.where((e==[0,0,0]).all(axis=2))] = [255,255,255];	