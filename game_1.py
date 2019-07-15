import numpy as np
import cv2
import time
from sklearn.cluster import KMeans
import pickle
from pprint import pprint
from sklearn.decomposition import PCA

n_cells = 2*2


#----------------GRID VERTICES IDENTIFICATION--------------------
#-----------------------------------------------------------------

'''
cap = cv2.VideoCapture(0)
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

corners = cv2.goodFeaturesToTrack(frame_bw,4,0.08,250)
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
THRESHOLD = 145
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
								'cluster_centroid':(int(coord[0]),int(coord[1])),
								'cluster_lbl':coord[4],
								'edge_rect':[coord[5],coord[6]]})

# dumping of cell_identification
pickle.dump(cell_identification, open('cell_identification.pickle','wb'))

pprint(cell_identification)

cv2.imshow('frame_warped',frame_warped)
cv2.imwrite('rectangle.png',frame_warped)
cv2.waitKey(0)
'''



#------------------------START---------------------------
#--------------------------------------------------------

THRESHOLD = 40



cell_identification = pickle.load(open('cell_identification.pickle','rb'))
grid_vertices = pickle.load(open('grid_vertices.pickle','rb'))
dataset = pickle.load(open('./dataset_moves/dataset.pickle','rb'))

X = [[c,x,y] for c,x,y,l in dataset]
y = [l for c,x,y,l in dataset]

X = np.asarray(X)

state = [0,0]



cap = cv2.VideoCapture(0)
time.sleep(2)
ret, background = cap.read()


background = cv2.flip(background,-1)
background = background[80:background.shape[0]-80,:]

pts1 = np.float32([grid_vertices[0],grid_vertices[1],grid_vertices[2],grid_vertices[3]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
background = cv2.warpPerspective(background,M,(300,300))
background = background[20:300-20,20:300-20]
game = background.copy()
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (5,5), 3)
cv2.imshow('frame',background)
cv2.waitKey(0)

#backsub = cv2.createBackgroundSubtractorMOG2()
counter = np.zeros((2,2))

while(True):

	ret, frame = cap.read()
	frame = cv2.flip(frame,-1)
	frame = frame[80:frame.shape[0]-80,:]
	pts1 = np.float32([grid_vertices[0],grid_vertices[1],grid_vertices[2],grid_vertices[3]])
	pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	frame = cv2.warpPerspective(frame,M,(300,300))
	frame = frame[20:300-20,20:300-20]
	frame_col = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.GaussianBlur(frame, (5,5), 3)
	#fgMask = backsub.apply(frame)
	sub = cv2.absdiff(frame,background)
	_,sub = cv2.threshold(sub, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

	print(state)

	# update counter
	for cell in cell_identification:
		ul=cell['edge_rect'][0]
		br=cell['edge_rect'][1]
		area_mean_intensity = np.mean(sub[ul[1]:br[1], ul[0]:br[0]])
		if area_mean_intensity != 255.0:
			counter[int(cell['id'][0]),int(cell['id'][1])] += 1
		else:
			counter[int(cell['id'][0]),int(cell['id'][1])] = 0
			action_performed = False
		
		print(cell['id']+'___'+str(area_mean_intensity))
	print()   


	# update rectangle and update state
	for cell in cell_identification:
		if counter[int(cell['id'][0]),int(cell['id'][1])] <= 3:
			cv2.rectangle(frame_col,cell['edge_rect'][0],cell['edge_rect'][1],(0,0,255),3)
		else:
			ul=cell['edge_rect'][0]
			br=cell['edge_rect'][1]
			cv2.rectangle(frame_col,ul,br,(0,255,0),3)
			current_cell = sub[ul[1]:br[1], ul[0]:br[0]]
			
			# corners
			corners = cv2.goodFeaturesToTrack(current_cell,10,0.1,10)

			# PCA
			points_coord = np.argwhere(current_cell==0)

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


			try:
				f = np.array([len(corners),x_pca,y_pca])
			except:
				print('Posiziona la card in maniera corretta')

			print(f)
		
			nn = np.argmin(np.sum(np.square(X-f),axis=1))
			print(y[nn])

			if not action_performed and state == [int(cell['id'][0]),int(cell['id'][1])]:
				action_performed = True
				current_cell = [int(cell['id'][0]),int(cell['id'][1])]

				if y[nn] == 'up' and state[1] != n_cells/2-1: state[1]+=1
				if y[nn] == 'down' and state[1] != 0: state[1]-=1
				if y[nn] == 'right' and state[0] != n_cells/2-1: state[0]+=1
				if y[nn] == 'left' and state[0] != 0: state[0]-=1


	# draw game
	for cell in cell_identification:
		if state == [int(cell['id'][0]),int(cell['id'][1])]:
			game_copy = game.copy()
			cv2.circle(game_copy,cell['cluster_centroid'],15,(0,0,255),-1)


	cv2.imshow('game',game_copy)
	cv2.imshow('sub',sub)
	cv2.imshow('frame_col',frame_col)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	time.sleep(1)


cap.release()
cv2.destroyAllWindows()
