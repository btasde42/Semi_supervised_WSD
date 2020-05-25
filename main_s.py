import structure_wsd_s
import argparse
import numpy as np
from collections import Counter, defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("vecteurs", help = "fichier vecteurs")
parser.add_argument("gold", help = "gold")
args = parser.parse_args()
with open(args.vecteurs) as file:
	file_vecteurs = file.readlines()
with open(args.gold) as gold:
	file_gold = gold.readlines()
vectors = []
for line in file_vecteurs:
	vector = [float(elt) for elt in line.split('\t')]
	vectors.append(vector)
vectors = np.array(vectors)
np.random.shuffle(vectors)

senses = Counter(int(line) for line in file_gold)
#print(senses)
classification = structure_wsd_s.KMeans(vectors, 6, None, "cosine") # 6 clusters pour abattre; à remplacer ensuite par le nombre de sens
#print(len(classification.examples))
clusters = classification.create_empty_clusters()
#print(len(classification.examples))
for i in range(3):
#	print("RUN ", i)
	for cluster in classification.clusters:
	distance = classification.distance_matrix("cosine")
	for j in range(len(distance.T)): # on parcourt les exemples ; il faut savoir à quel id des exemples correspond j
		min_value_index = np.argmin(distance.T[j]) # on trouve l'indice de la valeur min; c'est l'id du cluster
		#print(min_value_index)
		exo = classification.examples[j]
		classification.clusters[min_value_index].add_example_to_cluster(exo) # j = quel id de l'exemple ?
	for cluster in classification.clusters:
		#print("EXOS = ", len(classification.clusters[cluster].examples))
		#print("OLD center = ", classification.clusters[cluster].center)
		classification.clusters[cluster].recalculate_center()
		#print("NEW = ", classification.clusters[cluster].center)
		classification.clusters[cluster].delete_examples()
#################################
# 	print(len(distance.T))
# 	minValue = np.amin(distance)
# 	#print(minValue)
# 	# les coordonnées de la distance minimale dans la matrice
# 	# [i;j], i = cluster, j = numéro de l'exemple
# 	min_distance = np.where(distance == np.amin(distance))
# 	print(min_distance)
# 	#print('Tuple of arrays returned : ', min_distance)
# 	#print('List of coordinates of minimum value in Numpy array : ')
# 		 # zip the 2 arrays to get the exact coordinates
# 	listOfCordinates = list(zip(min_distance[0], min_distance[1]))
# 		 # travese over the list of cordinates
# 	print(listOfCordinates)
# 	for cord in listOfCordinates:
# 	#	print(cord)
# 	#	print(cord[1])
# 		# on regarde s'il y a des exemples qui se trouvent à la même distance de plusieurs clusters
# 		# on fait le choix du cluster en fonction de sa fréquence
# 		#same_examples = list(filter(lambda x: cord[1] in x, listOfCordinates))
# 		same_examples = [item for item in listOfCordinates if item[1]==cord[1]]
# 	#	print("Same examples ", same_examples)
# 		# si on l'exo se trouve à la même distance de plusieurs centres -> on choisit le cluster le plus fréquent
# 		if len(same_examples)>1:
# 			cluster =max([(elt[0], senses[elt[0]]) for elt in same_examples], key = lambda i : i[1])[0]
# 		else:
# 			cluster = cord[0]
# 		max(test_list, key = lambda i : i[1])[0] 
# 		print(clusters)

# 		#print(classification.examples[cord[1]])
# 		# il faut ajouter les exos au cluster et les supprimer des exos non-classifiés avec leur ID et pas avec la position !!!
# 		classification.clusters[cord[0]].add_example_to_cluster(classification.examples[cord[1]])
# 		classification.delete_example(classification.examples[cord[1]])
# 		#classification.delete_example(17)
# print(len(classification.examples))
# for cluster in classification.clusters:
# 	print("CLUSTER")
# 	print(cluster.examples)
	# ensuite il faut regarder les exemples à chaque coordonnée et ajouter l'exemple au cluster correspondant
	# il faut vérifier quoi faire si un exo est à la même distance de deux clusters
	# il faut calculer la distance par rapport au mean des exemples des clusters (centroids) et pas les vrais vecteurs
	