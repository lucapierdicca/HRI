import numpy as np
import cv2
import time

import pickle
from pprint import pprint
from sklearn.neighbors import NearestNeighbors

n_cells = 6*6
device = 0



print("STARTING CALIBRATION")

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

cap = cv2.VideoCapture(device + cv2.CAP_V4L)
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

cap = cv2.VideoCapture(device + cv2.CAP_V4L)
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




cap = cv2.VideoCapture(device + cv2.CAP_V4L)
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

	
	for i in cell_identification:
		cv2.rectangle(img_warped,
			i['edge_rect'][0], i['edge_rect'][1],
			(0,0,255),1)
		cv2.putText(img_warped,i['id'],i['cell_centroid'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
	
	cv2.imshow("img", img_warped)
    
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()


img_warped = cv2.imread('img_warped.png',1)
for i in cell_identification:
	cv2.rectangle(img_warped,
		i['edge_rect'][0], i['edge_rect'][1],
		(0,0,255),1)
	cv2.putText(img_warped,i['id'],i['cell_centroid'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

cv2.imshow('img_warped',img_warped)
cv2.waitKey(0)

print("CALIBRATION COMPLETE")