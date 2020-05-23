import structure_wsd_s
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("vecteurs", help = "fichier vecteurs")
args = parser.parse_args()
with open(args.vecteurs) as file:
	file_vecteurs = file.readlines()
vectors = []
for line in file_vecteurs:
	vector = [float(elt) for elt in line.split('\t')]
	vectors.append(vector)
vectors = np.array(vectors)
#print(vectors[:3])
np.random.shuffle(vectors)
#print(vectors[:3])

# comment savoir le nimbre de sens ?
#nn_senses = ?

classification = structure_wsd_s.KMeans(vectors, 6, None, "cosine") # 6 clusters pour abattre; à remplacer ensuite par le nombre de sens
print(len(classification.examples))
clusters = classification.create_empty_clusters()
for i,cluster in clusters.items():
	print(cluster.center)
print(len(classification.examples))
distance = classification.distance_matrix("cosine")
#print(len(distance))
#for i in range(classification.k):
#	for  j in range(len(classification.examples)):
#min_distance = np.where(distance == np.amin(distance))
#distance = np.array([[11, 12, 13],
 #                        [14, 15, 16],
  #                       [17, 15, 11],
   #                      [12, 14, 15]])
minValue = np.amin(distance)
print(minValue)
min_distance = np.where(distance == np.amin(distance))
#print('Tuple of arrays returned : ', min_distance)
print('List of coordinates of minimum value in Numpy array : ')
		 # zip the 2 arrays to get the exact coordinates
listOfCordinates = list(zip(min_distance[0], min_distance[1]))
		 # travese over the list of cordinates
for cord in listOfCordinates:
	print(cord)
	# ensuite il faut regarder les exemples à chaque coordonnée et ajouter l'exemple au cluster correspondant
	# il faut vérifier quoi faire si un exo est à la même distance de deux clusters