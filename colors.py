import numpy as np
import cv2

import time

colors = np.array([
		  [255,255,255],
		  [0,0,0],
		  [255,0,0],
		  [0,255,0],
		  [0,0,255],
		  [0,255,255],
		  [255,255,0],
		  [255,0,255]])

labels = ['bianco','nero','blu','verde','rosso','giallo','celeste','viola']

cap = cv2.VideoCapture(0)

while(True):

	ret, frame = cap.read()

	
	cv2.rectangle(frame,(270,190),(370,290),(0,0,255),3)
	frame_cut = frame[190:290,270:370,:]

	mean_color = np.mean(frame_cut, axis=(0,1))
	print(mean_color)


	cv2.imshow('frame',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


	print(labels[np.argmin(np.sum(np.square(colors - mean_color),axis=1))])


	time.sleep(0.5)
	#print(frame.shape)
	#print(frame)
	


cap.release()
cv2.destroyAllWindows()