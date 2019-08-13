import qi
import time
import cv2
import sys
import os
import sar_cmd


session = qi.Session()
ip = os.environ['PEPPER_IP']
port = '9559'


try:
        session.connect("tcp://" + ip + ":" + port)
except RuntimeError:
        print('error')
        sys.exit(1)

tabletService = session.service("ALTabletService")

#tabletService.resetTablet()
#tabletService.forgetWifi('SPQREL')

tabletService.disconnectWifi()
print(tabletService.configureWifi('wpa','SPQREL','robocup2017'))

print(tabletService.getWifiStatus())
tabletService.connectWifi('SPQREL')
time.sleep(2)
print(tabletService.getWifiStatus())
time.sleep(2)
print(tabletService.showWebview("http://10.0.1.203:8000"))         
