import sar_cmd
from sar_cmd import *
import time


posture_dict = {'left ear':[-0.9004466533660889, -0.20095157623291016, 0.0, 0.11811661720275879, -1.2225825786590576, -1.5002332925796509, -1.0876479148864746, 1.560058355331421, 
						-0.14266014099121094, 1.228718638420105, 0.5230875015258789, 0.010695934295654297],
			'right ear':[0.5997865200042725, -0.09970879554748535, 1.558524489402771, 0.14266014099121094, -1.2302525043487549, -0.5230875015258789, -0.0031099319458007812,
						 -0.09970879554748535, -0.10277676582336426, 1.398990511894226, 1.5017673969268799, 0.9863200187683105],
			'left eye':[0.0, -0.21015548706054688, -0.08743691444396973, 0.13499021530151367, -1.0032236576080322, -1.5477867126464844, 
						-1.4005842208862305, 1.560058355331421, -0.13959217071533203, 1.2195147275924683, 0.52001953125, 0.2868161201477051],
			'right eye':[0.0, -0.20095157623291016, 1.558524489402771, 0.14112615585327148, -1.2302525043487549, -0.5230875015258789, -0.015382051467895508,
						 -0.0720970630645752, -0.11351466178894043, 1.0047574043273926, 1.5355145931243896, 1.193410038948059],
			'head':[0.1994175910949707, 0.02454376220703125, -0.5522329807281494, 0.24697089195251465, -0.863631010055542, -1.1259419918060303, -1.5171680450439453, 1.6014759540557861, 
					-0.3006601333618164, 1.2210487127304077, 0.5246214866638184, 0.0367741584777832],
			'nose':[-0.0015339851379394531, -0.1994175910949707, 0.009203910827636719, 0.008726646192371845, -0.7992041110992432, -1.5539225339889526, -1.586197853088379, 1.5999419689178467, 
					-0.3006601333618164, 1.2210487127304077, 0.5230875015258789, 0.0367741584777832],
			'mouth':[0.0, -0.30833005905151367, 0.18867969512939453, 0.05062127113342285, -0.8114757537841797, -1.5508545637130737, -1.586197853088379, 1.5999419689178467, -0.3006601333618164,
					 1.2225826978683472, 0.5230875015258789, 0.0367741584777832],
			'right shoulder':[-0.9004466533660889, 0.3006601333618164, 1.558524489402771, 0.14266014099121094, -1.2302525043487549, -0.5230875015258789, 
								-0.009245872497558594, 0.7010293006896973, -0.18867969512939453, 1.5002332925796509, 1.5615923404693604, 1.4879380464553833],
			'left shoulder':[0.8007380962371826, 0.09970879554748535, 0.7010293006896973, 0.09664082527160645, -1.3008158206939697, -1.56159245967865, -1.3944478034973145,
							 1.556990385055542, -0.14572811126708984, 1.2210487127304077, 0.52001953125, 0.02143406867980957],
			'belly':[0.0, 0.5000777244567871, 1.0983302593231201, 0.1702718734741211, -0.6918253898620605, -1.164291501045227, 0.0014920234680175781, 1.556990385055542, -0.13959217071533203,
					 1.2195147275924683, 0.52001953125, 0.010695934295654297]}



if __name__ == "__main__":

	begin()

	pepper = sar_cmd.robot


	for k,v in posture_dict.items():
		pepper.say(k)
		pepper.setPosture(v)
		time.sleep(1)

	end()