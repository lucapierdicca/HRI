import qi
import time
import cv2
import sys

# genera randomicamente l'immagine del goal_p
def goal_p():
	bodyparts_to_coord = {'os':(304,65),
						  'od':(362,429)}

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

