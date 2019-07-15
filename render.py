import time
import cv2
import sys
from random import *

# genera randomicamente l'immagine del goal_p
def goal_p():
	bodyparts_to_coord = {'ors':(241,74),
						  'ord':(113,73),
						  'ocs':(153,84),
						  'ocd':(198,84)}

	bodyparts_sample = sample(bodyparts_to_coord.keys(),2)
	img = cv2.imread('./imgs/goal_p_half.png',1)

	for index,bp in enumerate(bodyparts_sample):
		cv2.circle(img,bodyparts_to_coord[bp],10,(255,0,0),-1)
		cv2.putText(img,str(index+1), 
			(bodyparts_to_coord[bp][0]-6,bodyparts_to_coord[bp][1]+5), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

	cv2.imwrite('./imgs/img.png',img)





'''
session = qi.Session()
ip = '10.0.1.202'
port = '9559'


try:
        session.connect("tcp://" + ip + ":" + port)
except RuntimeError:
        print('error')
        sys.exit(1)


tabletService = session.service("ALTabletService")



img = cv2.imread('grid_filled.png',1)
cv2.imwrite('grid_filled_m.png',img)
img = cv2.imread('grid_filled_m.png',1)

print(tabletService.showWebview("http://10.0.1.204:8000"))

time.sleep(1)

cv2.circle(img,(50,50), 6, (0,0,255), -1)
cv2.imwrite('grid_filled_m.png',img)
print(tabletService.showWebview("http://10.0.1.204:8000"))

time.sleep(1)


cv2.circle(img,(200,200), 6, (0,0,255), -1)
cv2.imwrite('grid_filled_m.png',img)
print(tabletService.showWebview("http://10.0.1.204:8000"))

'''

