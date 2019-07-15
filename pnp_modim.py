import os, sys, time

pdir = os.getenv('PNP_HOME')
sys.path.insert(0, pdir+'/PNPnaoqi/py')

import pnp_cmd_naoqi
from pnp_cmd_naoqi import *

from pprint import pprint 



if __name__ == "__main__":


	p = PNPCmd()

	p.begin()


	p.exec_action('Show', '')

	p.exec_action('Say', 'Hi')

	




	p.end()