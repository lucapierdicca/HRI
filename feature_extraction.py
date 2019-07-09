import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
import pickle
from pprint import pprint

def get_rotation(theta):
	theta = np.radians(theta)
	R = np.array([[np.cos(theta), -np.sin(theta)],
				  [np.sin(theta), np.cos(theta)]],dtype=int)
	return R


move_imgs_dir = './dataset_moves/imgs/'
img_list = os.listdir(move_imgs_dir)


dataset = []

for imgfile in img_list:
	print(imgfile)
	dataset.append([])

	img = cv2.imread(move_imgs_dir+imgfile,0)

	# orientation (PCA)
	points_coord = np.argwhere(img==0)

	x_prime = points_coord[:,1]
	y_prime = 650-points_coord[:,0]

	x_prime_mean = round(np.mean(x_prime))
	y_prime_mean = round(np.mean(y_prime))

	x_prime = x_prime - x_prime_mean
	y_prime = y_prime - y_prime_mean

	points_coord_prime = np.vstack((x_prime,y_prime))


	points_coord_prime = points_coord_prime.T

	pca = PCA(n_components=1, svd_solver='full')
	pca.fit(points_coord_prime) 
	x_pca = int(round(pca.components_[0][0]))
	y_pca = int(round(pca.components_[0][1]))
	print(x_pca, y_pca)

	dataset[-1].insert(0,x_pca)
	dataset[-1].append(y_pca)




	# edges
	edges = cv2.Canny(img,650,650)
	edges_coord = np.argwhere(edges==255)

	y_prime = 650-edges_coord[:,0]
	x_prime = edges_coord[:,1]

	x_prime = x_prime - x_prime_mean
	y_prime = y_prime - y_prime_mean

	edges_coord_prime = np.vstack((x_prime,y_prime))

	edges_coord_prime = edges_coord_prime.T

	print(edges_coord_prime.shape)




	# corners (n°)
	corners = cv2.goodFeaturesToTrack(img,10,0.1,10)

	dataset[-1].insert(0,len(corners))
	dataset[-1].append(imgfile[:imgfile.find('.')])


pprint(dataset)
pickle.dump(dataset, open('./dataset_moves/dataset.pickle','wb'))


'''
# take a look
plt.scatter(points_coord_prime[:,0], points_coord_prime[:,1], s=0.002)
plt.scatter(edges_coord_prime[:,0], edges_coord_prime[:,1], c='green', s=0.5)
plt.scatter(pca.components_[0][0],pca.components_[0][1], c='red', s=0.2)
plt.scatter(0,0,c='black', s=0.5)
plt.ylim(-325,325)
plt.xlim(-325,325)

plt.show()




# haussdorff
from scipy.spatial.distance import directed_hausdorff
h = max(directed_hausdorff(edges_coord_prime, edges_coord_prime)[0], 
	    directed_hausdorff(edges_coord_prime, edges_coord_prime)[0])

print(h)
'''