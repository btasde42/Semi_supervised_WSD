
from create_vectors import *
from structure_wsd_s import *
import argparse
import numpy as np
import random
from scipy.spatial.distance import cosine

parser = argparse.ArgumentParser()
parser.add_argument("verbe", help = "abattre, aborder, affecter, comprendre, compter")
parser.add_argument('conll', help = 'le fichier conll full path')
parser.add_argument('gold', help = 'le fichier classe gold full path')
parser.add_argument('tok_ids', help = 'fichier tokens ids full path')
parser.add_argument("inventaire", help = 'inventaire de sens full path')
parser.add_argument("--r",help='Y ou N selon si on veut reduire les vecteurs, IMPORTANT: #si on applique pas la reduction, fusion_method doit etre moyenne ou concat')
parser.add_argument("--traits",nargs='+',help="List des traits qu'on veut utiliser. [syntx,linear]")
parser.add_argument("--n",type=int, help='la taille de contexte pour les linears. Optionelle.')
parser.add_argument("--fusion_method",help="La methode de fusion pour differents types des vecteurs de traits s'il y en a plusieurs")
parser.add_argument("--linear_method", help='Concat, somme or moyenne pour fusionner les traits de linear')
parser.add_argument("--dim",help="La taille de dimention reduit pour les vecteurs de verbe")


args = parser.parse_args()

verbes = ["abattre", "aborder", "affecter", "comprendre", "compter"]
# on s'assure que les fichiers correspondent au verbe sélectionné
assert args.verbe in verbes
assert args.verbe in args.conll 
assert args.verbe in args.gold
assert args.verbe in args.tok_ids



with open(args.conll) as file:
	file_conll = file.read()
with open(args.gold) as file2:
	file_gold = file2.readlines()
with open(args.tok_ids) as file3:
	file_ids = file3.readlines()


vectors_syntx,num_senses,vectors_linear=read_conll(file_conll, file_gold, file_ids, args.n,args.inventaire,args.linear_method)



if len(args.traits)==1: #s'il y a pas les deux traits démandé mais qu'un seul
	if args.traits[0].lower() =='syntx':
		if args.r.lower()=='y': #si la reduction est demandé
			np.savetxt(args.verbe+"_vectors_syntx", vectors_syntx, delimiter = "\t")
			vectors_syntx=reduce_dimension(vectors_syntx,'syntx',args.verbe,int(args.dim))
		else: 
			vectors_syntx=vectors_syntx
			
		examples=Examples()
		for i in range(len(vectors_syntx)):
			gold=file_gold[i]
			vector=Ovector(i,gold,None,vectors_syntx[i],None)
			vector.set_vector(vectors_syntx[i])
			examples.set_vector_to_matrix(vector)
			

	elif args.traits[0].lower() =='linear':
		if args.r.lower()=='y':
			np.savetxt(args.verbe+"_linear_vectors", vectors_linear, delimiter = "\t")
			vectors_linear=reduce_dimension(vectors_linear,'linear',args.verbe,int(args.dim))

		else:
			vectors_linear=vectors_linear

		examples=Examples()
		for i in range(len(vectors_linear)):
			vector=vectors_linear[i]
			gold=file_gold[i]
			vector=Ovector(i,gold,None,None,vectors_linear[i])
			vector.set_vector(vectors_linear[i])
			examples.set_vector_to_matrix(vector)
	else:
		print("Traits démandée n'existe pas!")		

else: #si on demande tous les deux traits linear et syntx
	if args.r.lower()=='y': #si la reduction est demandé
		np.savetxt(args.verbe+"_vectors_syntx", vectors_syntx, delimiter = "\t")
		np.savetxt(args.verbe+"_linear_vectors", vectors_linear, delimiter = "\t")
		vectors_linear=reduce_dimension(vectors_linear,'linear',args.verbe,int(args.dim))
		vectors_syntx=reduce_dimension(vectors_syntx,'syntx',args.verbe,int(args.dim))
		examples=Examples() 
		for i in range(len(vectors_linear)):
			gold=file_gold[i]
			vector=Ovector(i,gold,args.fusion_method,vectors_syntx[i],vectors_linear[i])
			vector.fusion_traits()
			examples.set_vector_to_matrix(vector)

	else:
		examples=Examples()
		for i in range(len(vectors_linear)):
			gold=file_gold[i]
			vector=Ovector(i,gold,args.fusion_method,vectors_syntx[i],vectors_linear[i])
			vector.fusion_traits()
			examples.set_vector_to_matrix(vector)


espace_vectorielle=examples.get_espace_vec()


# for i in espace_vectorielle:
# 	print(i.get_index())
# 	print(i.get_vector())
# 	print(i.get_gold_class())
# 	print('\t')

E = 10 # nombre d'époques pour tourner l'algo

matrix=examples.get_espace_vec()
print(type(matrix))
random.shuffle(matrix)
senses = Counter(int(line) for line in file_gold)
print("SENSES ", senses)
N = len(senses.keys()) # le nb de clusters souhaité
GOLD = senses.keys() # les numéros des sens, les classes gold
classification = KMeans(examples.espace_vec, N, GOLD, None, "cosine")
clusters = classification.create_empty_clusters()
# variante kmeans 1
for i in range(E):
	for cluster_id in classification.clusters:
			classification.clusters[cluster_id].delete_examples()
	for exo in classification.examples:
		distances = []
		for cluster_id in classification.clusters:
			if exo != classification.clusters[cluster_id].initial_example:
				distances.append(cosine(exo.vector, classification.clusters[cluster_id].center))
		minimum_distance = np.argmin(distances)
		classification.clusters[minimum_distance].add_example_to_cluster(exo)
	for cluster_id in classification.clusters:
		classification.clusters[cluster_id].recalculate_center()
print("RESULTS 1 : ")
for i in classification.clusters:
	print("CLUSTER ", i)
	print(len(classification.clusters[i].examples))
	print(classification.clusters[i].id)
	print(Counter([exo.gold for exo in classification.clusters[i].examples]))

# variante kmeans 2
for i in range(E):
	distance = classification.distance_matrix("cosine")
	for j in range(len(distance.T)): # on parcourt les exemples ; il faut savoir à quel id des exemples correspond j
		min_value_index = np.argmin(distance.T[j]) # on trouve l'indice de la valeur min; c'est l'id du CLUSTER
		exo = matrix[j] # exo à ajouter dans le cluster #min_value_index
		if exo != classification.clusters[min_value_index].initial_example :
			classification.clusters[min_value_index].add_example_to_cluster(exo) # j = quel id de l'exemple ?
	for cluster in classification.clusters:
		classification.clusters[cluster].recalculate_center()
		if i < E-1 : # on va supprimer les exemples des clusters jusqu'au dernier run
			classification.clusters[cluster].delete_examples()
print("RESULTS 2 : ")
for i in classification.clusters:
	print("CLUSTER ", i)
	print(len(classification.clusters[i].examples))
	print(classification.clusters[i].id)
	print(Counter([exo.gold for exo in classification.clusters[i].examples]))