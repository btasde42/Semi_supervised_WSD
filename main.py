import datetime
import csv
from create_vectors import *
from structure_wsd_s import *
import argparse
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean, cityblock


parser = argparse.ArgumentParser()
parser.add_argument("verbe", help = "abattre, aborder, affecter, comprendre, compter")
parser.add_argument('conll', help = 'le fichier conll full path')
parser.add_argument('gold', help = 'le fichier classe gold full path')
parser.add_argument('tok_ids', help = 'fichier tokens ids full path')
parser.add_argument("inventaire", help = 'inventaire de sens full path')
parser.add_argument("--dist_formula", help='Name a distance formula between cosine / euclidean/ cityblock (==manatthan)')
parser.add_argument("--r",help='Y ou N selon si on veut reduire les vecteurs, IMPORTANT: #si on applique pas la reduction, fusion_method doit etre moyenne ou concat')
parser.add_argument("--traits",nargs='+',help="List des traits qu'on veut utiliser. [syntx,linear]")
parser.add_argument("--n",type=int, help='la taille de contexte pour les linears. Optionelle.')
parser.add_argument("--fusion_method",help="La methode de fusion pour differents types des vecteurs de traits s'il y en a plusieurs")
parser.add_argument("--linear_method", help='somme ou moyenne pour fusionner les traits de linear')
parser.add_argument("--dim",help="La taille de dimention reduit pour les vecteurs de verbe")
parser.add_argument("--cluster_type",help="le type de clustering kmeans : basic, constrained, ++, constrained++ ")

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

folder="Essaies_"+args.verbe+'_'+datetime.datetime.now().strftime("%I:%M%p - %B %d, %Y")
os.makedirs(folder, exist_ok=True) #on cree un dossier pour chque essaie afin d'enregistrer les essaies
csv_file="{}.csv".format(folder+'/'+ "ARGS") #ECRITURE DES RESULTATS
with open(csv_file, 'w') as outfile:
	w = csv.DictWriter(outfile, vars(args).keys())
	w.writeheader()
	w.writerow(vars(args))

vectors_syntx,num_senses,vectors_linear=read_conll(file_conll, file_gold, file_ids, args.n,args.inventaire,args.linear_method)



if len(args.traits)==1: #s'il y a pas les deux traits démandé mais qu'un seul
	if args.traits[0].lower() =='syntx':
		if args.r.lower()=='y': #si la reduction est demandé
			np.savetxt("{}.txt".format(folder+'/'+args.verbe+"_vectors_syntx"), vectors_syntx, delimiter = "\t")
			vectors_syntx=reduce_dimension(vectors_syntx,'syntx',args.verbe,int(args.dim))
		else: 
			vectors_syntx=vectors_syntx
			
		examples=Examples()
		for i in range(len(vectors_syntx)):
			gold=file_gold[i].strip('\n')
			vector=Ovector(i,gold,None,vectors_syntx[i],None)
			vector.set_vector(vectors_syntx[i])
			examples.set_vector_to_matrix(vector)
			

	elif args.traits[0].lower() =='linear':
		if args.r.lower()=='y':
			np.savetxt("{}.txt".format(folder+'/'+args.verbe+"_linear_vectors"), vectors_linear, delimiter = "\t")
			vectors_linear=reduce_dimension(vectors_linear,'linear',args.verbe,int(args.dim))

		else:
			vectors_linear=vectors_linear

		examples=Examples()
		for i in range(len(vectors_linear)):
			vector=vectors_linear[i]
			gold=file_gold[i].strip('\n')
			vector=Ovector(i,gold,None,None,vectors_linear[i])
			vector.set_vector(vectors_linear[i])
			examples.set_vector_to_matrix(vector)
	else:
		print("Traits démandée n'existe pas!")		

else: #si on demande tous les deux traits linear et syntx
	if args.r.lower()=='y': #si la reduction est demandé
		np.savetxt("{}.txt".format(folder+'/'+args.verbe+"_vectors_syntx"), vectors_syntx, delimiter = "\t")
		np.savetxt("{}.txt".format(folder+'/'+args.verbe+"_linear_vectors"), vectors_linear, delimiter = "\t")
		vectors_linear=reduce_dimension(vectors_linear,'linear',args.verbe,int(args.dim))
		vectors_syntx=reduce_dimension(vectors_syntx,'syntx',args.verbe,int(args.dim))
		examples=Examples() 
		#print("LEN LINEAR : ", len(vectors_linear))
		for i in range(len(vectors_linear)):
			gold=file_gold[i].strip('\n')
			vector=Ovector(i,gold,args.fusion_method,vectors_syntx[i],vectors_linear[i])
			vector.fusion_traits()
			examples.set_vector_to_matrix(vector)
		print("TRY EXAMPLES 125 : ", examples.get_Ovector_by_id(125).vector)
	else:
		examples=Examples()
		for i in range(len(vectors_linear)):
			gold=file_gold[i].strip('\n')
			vector=Ovector(i,gold,args.fusion_method,vectors_syntx[i],vectors_linear[i])
			vector.fusion_traits()
			examples.set_vector_to_matrix(vector)


E = 50# nombre d'époques pour tourner l'algo



matrix=examples.get_espace_vec()

rd.shuffle(matrix)
senses = Counter(int(line) for line in file_gold)
print("SENSES ", senses)
N = len(senses.keys()) # le nb de clusters souhaité
GOLD = senses.keys() # les numéros des sens, les classes gold
print(N)

###INITIALISATION DES CENTRES AVEC KMEANS++####
if args.cluster_type.lower() =='++' :
	centers=[]
	i=rd.randint(0,len(matrix))
	print("I = ", i)
	center= examples.get_Ovector_by_id(i) #on rajoute une premier vector aléatoire comme centre
	#print("FIRST CENTER = ", center.vector)
	centers.append(center)
	for cluster in range(N-1): #on va initialiser le k-1 autre centre de cluster qui reste
		distance=np.array([])
		for i in matrix: #chaque exemple dans l'espace
			point=i.vector
			distance=np.append(distance,np.min(np.sum((point-center.vector)**2)))
		
		proba=distance/np.sum(distance)
		cumul_proba=np.cumsum(proba)
		r=rd.random()
		i=0
		for j,p in enumerate(cumul_proba):
			if r<p:
				i=j
				break
		centers.append(examples.get_Ovector_by_id(i))

	print("CENTERS")
	print([(c.gold,c.vector) for c in centers])
	classification1 = KMeans(matrix, N, GOLD, None, args.dist_formula ,centers)
	classification1.create_empty_clustersPlus() #KMEANS ++

	classification2 = KMeans(matrix, N, GOLD, None, args.dist_formula ,centers)
	classification2.create_empty_clustersPlus() #KMEANS ++

### CONSTRAINED ++ ######
if args.cluster_type.lower() =='constrained++':
	centers=[]
	centers_gold=[]
	i=rd.randint(0,len(matrix))
	print("I = ", i)
	center= examples.get_Ovector_by_id(i) #on rajoute une premier vector aléatoire comme centre
	centers_gold.append(examples.get_Ovector_by_id(i).gold)
	#print("FIRST CENTER = ", center.vector)
	centers.append(center)
	for cluster in range(N-1): #on va initialiser le k-1 autre centre de cluster qui reste
		distance=np.array([])
		for x in matrix: #chaque exemple dans l'espace
			point=x.vector
			distance=np.append(distance,np.min(np.sum((point-center.vector)**2)))
		
		proba=distance/np.sum(distance)
		cumul_proba=np.cumsum(proba)
		r=rd.random()
		i=0
		for j,p in enumerate(cumul_proba):
			if r<p and examples.get_Ovector_by_id(j).gold not in centers_gold:
				i=j
				break
		centers.append(examples.get_Ovector_by_id(i))
		centers_gold.append(examples.get_Ovector_by_id(i).gold)

	print("CENTERS")
	print([(c.gold,c.vector) for c in centers])
	classification1 = KMeans(matrix, N, GOLD, None, args.dist_formula ,centers)
	classification1.create_empty_clustersPlus() #KMEANS ++

	classification2 = KMeans(matrix, N, GOLD, None, args.dist_formula ,centers)
	classification2.create_empty_clustersPlus() #KMEANS ++

########################
if args.cluster_type.lower() == 'constrained' or 'constrained++':
	classification1 = KMeans(matrix, N, GOLD, None, args.dist_formula, None) # on n'a pas encore de centres
	classification1.create_empty_clusters() # constrainted Kmeans
	
	classification2 = KMeans(matrix, N, GOLD, None, args.dist_formula, None) # on n'a pas encore de centres
	classification2.create_empty_clusters() # constrainted Kmeans

	for i in range(E):
		for cluster_id in classification1.clusters:
				classification1.clusters[cluster_id].delete_examples()
				classification1.clusters[cluster_id].resave_initial_example()
				#print(classification1.clusters[cluster_id].initial_example)
		for exo in classification1.examples:
			distances = []
			for cluster_id in classification1.clusters:
				if exo != classification1.clusters[cluster_id].initial_example:
					# print(type(exo.vector))
					# print(exo.vector)
					# print(type(classification1.clusters[cluster_id].center))
					# print(type(classification1.clusters[cluster_id].center.vector))
					# print(classification1.clusters[cluster_id].center.vector)
					distances.append(cosine(exo.vector, classification1.clusters[cluster_id].center))
			minimum_distance = np.argmin(distances)
			classification1.clusters[minimum_distance].add_example_to_cluster(exo)
		for cluster_id in classification1.clusters:
			#print("INITIAL : ", classification1.clusters[cluster_id].initial_example)
			classification1.clusters[cluster_id].recalculate_center()
			#print("NEW CENTER ", type(classification1.clusters[cluster_id].center))
			#print(classification1.clusters[cluster_id].center.vector)

	print("RESULTS 1 : ")
	cluster_dict1={}
	for i in classification1.clusters:
		classif1=Counter([exo.gold for exo in classification1.clusters[i].examples])
		classification1.clusters[i].redefine_id(max(classif1,key=classif1.get)) #id de cluster == la classe le plus nombreaux
		cluster_dict1[classification1.clusters[i].id]=classif1
		print("CLUSTER ", i)
		print(len(classification1.clusters[i].examples))
		print(classification1.clusters[i].id)
		print(classif1)
		print('\t')
	print(cluster_dict1)

	eval1=evaluate(classification1.clusters)

	print(eval1)

	csv_file="{}.csv".format(folder+'/'+ "KMEANS1") #ECRITURE DES RESULTATS
	dfa = pd.DataFrame(cluster_dict1)
	dfa.to_csv(csv_file)

	csv_file="{}.csv".format(folder+'/'+ "KMEANS1_evaluate") #ECRITURE DES RESULTATS
	dfb = pd.DataFrame(eval1)
	dfb.to_csv(csv_file)



	# variante kmeans 2
	for i in range(E):
		distance = classification2.distance_matrix(args.dist_formula)
		for j in range(len(distance.T)): # on parcourt les exemples ; il faut savoir à quel id des exemples correspond j
			min_value_index = np.argmin(distance.T[j]) # on trouve l'indice de la valeur min; c'est l'id du CLUSTER
			exo = matrix[j] # exo à ajouter dans le cluster #min_value_index
			if exo != classification2.clusters[min_value_index].initial_example :
				classification2.clusters[min_value_index].add_example_to_cluster(exo) # j = quel id de l'exemple ?
		for cluster in classification2.clusters:
			classification2.clusters[cluster].recalculate_center()

			if i < E-1 : # on va supprimer les exemples des clusters jusqu'au dernier run
				classification2.clusters[cluster].delete_examples()
				#classification2.clusters[cluster].resave_initial_example()

	print("RESULTS 2 : ")
	cluster_dict2={}
	for i in classification2.clusters:
		classif2=Counter([exo.gold for exo in classification2.clusters[i].examples])
		classification2.clusters[i].redefine_id(max(classif2,key=classif2.get)) #id de cluster == la classe le plus nombreaux
		cluster_dict2[classification2.clusters[i].id]=classif2
		print("CLUSTER ", i)
		print(len(classification2.clusters[i].examples))
		print(classification2.clusters[i].id)
		print(classif2)
		print('\t')
	print(cluster_dict2)

	eval2=evaluate(classification2.clusters)
	print(eval2)


if args.cluster_type.lower() == "++":
	for i in range(E):
		for cluster_id in classification1.clusters:
				classification1.clusters[cluster_id].delete_examples()
				#if args.cluster_type == "constrained" :
				#	classification1.clusters[cluster_id].resave_initial_example()
		for exo in classification1.examples:
			distances = []
			for cluster_id in classification1.clusters:
				#print("INITIAL : ", classification1.clusters[cluster_id].initial_example)
				if exo != classification1.clusters[cluster_id].initial_example:
					if args.dist_formula.lower() == 'cosine':
						distances.append(cosine(exo.vector, classification1.clusters[cluster_id].center))
					elif args.dist_formula.lower() == 'euclidean':
						distances.append(euclidean(exo.vector, classification1.clusters[cluster_id].center))
					else:
						distances.append(cityblock(exo.vector, classification1.clusters[cluster_id].center))
			minimum_distance = np.argmin(distances)
			classification1.clusters[minimum_distance].add_example_to_cluster(exo)
		for cluster_id in classification1.clusters:
			classification1.clusters[cluster_id].recalculate_center()
		#classification1.clusters[cluster].resave_initial_example()


	print("RESULTS 1 : ")
	cluster_dict1={}
	for i in classification1.clusters:
		classif1=Counter([exo.gold for exo in classification1.clusters[i].examples])
		classification1.clusters[i].redefine_id(max(classif1,key=classif1.get)) #id de cluster == la classe le plus nombreaux
		cluster_dict1[classification1.clusters[i].id]=classif1
		print("CLUSTER ", i)
		print(len(classification1.clusters[i].examples))
		print(classification1.clusters[i].id)
		print(classif1)
		print('\t')
	print(cluster_dict1)

	eval1=evaluate(classification1.clusters)

	print(eval1)

	csv_file="{}.csv".format(folder+'/'+ "KMEANS1") #ECRITURE DES RESULTATS
	dfa = pd.DataFrame(cluster_dict1)
	dfa.to_csv(csv_file)

	csv_file="{}.csv".format(folder+'/'+ "KMEANS1_evaluate") #ECRITURE DES RESULTATS
	dfb = pd.DataFrame(eval1)
	dfb.to_csv(csv_file)



	# variante kmeans 2
	for i in range(E):
		distance = classification2.distance_matrix(args.dist_formula)
		for j in range(len(distance.T)): # on parcourt les exemples ; il faut savoir à quel id des exemples correspond j
			min_value_index = np.argmin(distance.T[j]) # on trouve l'indice de la valeur min; c'est l'id du CLUSTER
			exo = matrix[j] # exo à ajouter dans le cluster #min_value_index
			if exo != classification2.clusters[min_value_index].initial_example :
				classification2.clusters[min_value_index].add_example_to_cluster(exo) # j = quel id de l'exemple ?
		for cluster in classification2.clusters:
			classification2.clusters[cluster].recalculate_center()

			if i < E-1 : # on va supprimer les exemples des clusters jusqu'au dernier run
				classification2.clusters[cluster].delete_examples()
				#classification2.clusters[cluster].resave_initial_example()

	print("RESULTS 2 : ")
	cluster_dict2={}
	for i in classification2.clusters:
		classif2=Counter([exo.gold for exo in classification2.clusters[i].examples])
		classification2.clusters[i].redefine_id(max(classif2,key=classif2.get)) #id de cluster == la classe le plus nombreaux
		cluster_dict2[classification2.clusters[i].id]=classif2
		print("CLUSTER ", i)
		print(len(classification2.clusters[i].examples))
		print(classification2.clusters[i].id)
		print(classif2)
		print('\t')
	print(cluster_dict2)

	eval2=evaluate(classification2.clusters)
	print(eval2)

csv_file="{}.csv".format(folder+'/'+ "KMEANS2") #ECRITURE DES RESULTATS
dfc = pd.DataFrame(cluster_dict2)
dfc.to_csv(csv_file)

csv_file="{}.csv".format(folder+'/'+ "KMEANS2_evaluate") #ECRITURE DES RESULTATS
dfd = pd.DataFrame(eval2)
dfd.to_csv(csv_file)


##########DATA VISUALISATION ############
plt.rc('font', size=16)

#set style of plots
sns.set_style('white')

#define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)



