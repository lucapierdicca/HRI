import os, sys, time

pdir = os.getenv('PNP_HOME')
sys.path.insert(0, pdir+'/PNPnaoqi/py')

import pnp_cmd_naoqi
from pnp_cmd_naoqi import *

import render


if __name__ == "__main__":


	p = PNPCmd()

	p.begin()

	# 1. presentazione


	# 2. presentazione goal esercizio
	render.goal_p()

	p.exec_action('Show', '')
	p.exec_action('Say', 'Hi_can_you_help_me?')


	# 3. svolgimento esercizio guidato

	# 4. esecuzione plan_tessere

	# 5. gioco

	# 6. saluto


	p.end()