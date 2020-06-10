
from create_vectors import *
from structure_wsd import *
import argparse
import numpy as np

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


for i in espace_vectorielle:
	print(i.get_index())
	print(i.get_vector())
	print(i.get_gold_class())
	print('\t')


classification = structure_wsd_s.KMeans(vectors, num_senses, None, "cosine") # 6 clusters pour abattre; à remplacer ensuite par le nombre de sens
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

