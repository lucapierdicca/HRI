import os, sys, time

pdir = os.getenv('PNP_HOME')
sys.path.insert(0, pdir+'/PNPnaoqi/py')

import pnp_cmd_naoqi
from pnp_cmd_naoqi import *

from pprint import pprint 


def checkConditions(p):	
	touch_service = p.app.session.service('ALTouch')
	while True:
		print('check')
		status = touch_service.getStatus()
		pprint(status)
		if status[1][1] == True:
			print('larm')
			p.set_condition('beentouched', True)
			p.set_condition('larmtouched', True)
			
		elif status[3][1] == True:
			print('rarm')
			p.set_condition('beentouched', True)
			p.set_condition('rarmtouched', True)
			

		time.sleep(0.5)




# Start action server
if __name__ == "__main__":


	p = PNPCmd()

	p.begin()


	r = threading.Thread(target=checkConditions, args=[p])
	r.start()

    #p.start_action('L','')
    #p.start_action('R','')
    #p.start_action('S',"Turning_my_wrists")

	while True:
		p.set_condition('larmtouched', False)
		p.set_condition('rarmtouched', False)
		p.set_condition('beentouched', False)

		p.exec_action('Wait', '', interrupt='beentouched')

		if p.get_condition('larmtouched') == True:
			p.exec_action('Rarm', '0.50_-0.26_1.30_2.4_-1.08')
		elif p.get_condition('rarmtouched') == True:
			p.exec_action('Larm', '-0.50_0.26_-1.30_-2.4_1.08')

		#p.start_action('Say', 'Ciao')
		#p.exec_action('Rwrist', '1.08')


		# wave, say = 'run', 'run'
		# while (wave == 'run' or say == 'run'):
		# 	wave = p.action_status('Wave')
		# 	say = p.action_status('Say')
		# 	print(wave,say)
		# 	time.sleep(0.2) 


		p.exec_action('normPosture','')

	#p.execute_plan('saluto')

    
    #p.exec_action('wait', '5', interrupt='timeout_2.5', recovery='wait_3;skip_action')  # blocking

    #p.exec_action('wait', '5', interrupt='mycondition', recovery='wait_3;skip_action')  # blocking


	p.end()