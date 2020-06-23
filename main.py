import sys
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
from itertools import product, chain

######CREER HYPERPARAMETRE COMBINATIONS########
params_grid={'--dist_formula':["euclidean"],'--r': ["y","n"],'--traits':['syntx','deux'],
'--n':['2','3'],'--fusion_method':["moyenne","concat","somme"],'--linear_method':["moyenne","concat","somme"],'--dim':['5'],'--tfidf':["y","n"]}

param_combinations=[]
for vals in product(*params_grid.values()):
	if vals[1].lower()=='n' and vals[4].lower()=='somme':
		continue
	else:
		nested=list([list(a) for a in zip(params_grid, vals)])
		flat=list(chain.from_iterable(nested))
		param_combinations.append(flat)

#print(len(param_combinations))
####################################################

obligparser = argparse.ArgumentParser() #obligatory terminal based arguments
elseparser= argparse.ArgumentParser() #param combination based arguments

obligparser.add_argument("verbe", help = "abattre, aborder, affecter, comprendre, compter")
obligparser.add_argument("cluster_type", choices=('kmeans++','constrained','constrained++'), help="le type de clustering kmeans : constrained, kmeans++, constrained++ ")

elseparser.add_argument("--dist_formula", choices=('euclidean','cosine','cityblock'),help='Name a distance formula between cosine / euclidean/ cityblock (==manatthan)')
elseparser.add_argument("--r", choices=('y','n'),help='Y ou N selon si on veut reduire les vecteurs, IMPORTANT: #si on applique pas la reduction, fusion_method doit etre moyenne ou concat')
elseparser.add_argument("--traits",help="List des traits qu'on veut utiliser. [syntx,linear,deux]")
elseparser.add_argument("--n",type=int, choices=(2,3,4),help='la taille de contexte pour les linears. Optionelle.')
elseparser.add_argument("--fusion_method", choices=('moyenne','concat','somme'),help="La methode de fusion pour differents types des vecteurs de traits s'il y en a plusieurs")
elseparser.add_argument("--linear_method", choices=('moyenne','concat','somme'), help='somme ou moyenne pour fusionner les traits de linear')
elseparser.add_argument("--dim",type=int,choices=(5,6,7),help="La taille de dimention reduit pour les vecteurs de verbe")
elseparser.add_argument("--tfidf",choices=('y','n'), help="la pondération des mots du contextes : y / n")

#'--verbe':['abattre','aborder','affecter','comprendre','compter'],
#args=parser.parse_args()
argo=obligparser.parse_args()

all_results=[] #result list pour la fin

for i in param_combinations:
	args=elseparser.parse_args(i)

	print(vars(args))
	conlls=["data/abattre/abattre-161.conll","data/aborder/aborder-221.conll","data/affecter/affecter-191.conll","data/comprendre/comprendre-150.conll","data/compter/compter-150.conll"]
	golds=["data/abattre/abattre-161.gold","data/aborder/aborder-221.gold","data/affecter/affecter-191.gold","data/comprendre/comprendre-150.gold","data/compter/compter-150.gold"]
	tok_ids=["data/abattre/abattre-161.tok_ids","data/aborder/aborder-221.tok_ids","data/affecter/affecter-191.tok_ids","data/comprendre/comprendre-150.tok_ids","data/compter/compter-150.tok_ids"]
	inventaire="inventaire_de_sens.xlsx"

	if argo.verbe=="abattre":
		conll=conlls[0]
		gold=golds[0]
		tok_id=tok_ids[0]
	if argo.verbe == "aborder":
		conll=conlls[1]
		gold=golds[1]
		tok_id=tok_ids[1]
	if argo.verbe == "affecter":
		conll=conlls[2]
		gold=golds[2]
		tok_id=tok_ids[2]

	if argo.verbe == "comprendre":
		conll=conlls[3]
		gold=golds[3]
		tok_id=tok_ids[3]

	if argo.verbe == "compter":
		conll=conlls[4]
		gold=golds[4]
		tok_id=tok_ids[4]


	########## GESTION EXCEPTION ##########

	verbes = ["abattre", "aborder", "affecter", "comprendre", "compter"]
	possible_fusion = ["somme", "moyenne", "concat"]
	possible_clusters = ["constrained", "kmeans++", "constrained++"]
	possible_formula = ["cosine", "euclidean", "cityblock"]
	yn = ["y", "n"]

	assert argo.verbe in verbes, "verbe inconnu"
	assert argo.verbe in conll, "verbe ou fichier conll incorrect"
	assert argo.verbe in gold, "verbe ou fichier gold incorrect"
	assert argo.verbe in tok_id, "verbe ou fichier tok_ids incorrect"
	if len(args.traits) == 2:
		assert args.fusion_method in possible_fusion, "méthode de fusion incorrecte, choisissez parmi : somme, moyenne, concat"
	assert args.linear_method in possible_fusion, "méthode de fusion incorrecte, choisissez parmi : somme, moyenne, concat"
	assert args.r in yn, "choisissez s'il faut faire la réduction ou non : --r y / n"
	assert args.tfidf, "choisissez s'il fait faire la pondération tfidf : --tfidf y / n"
	assert argo.cluster_type in possible_clusters, "type de clustering incorrect, choisissez parmi : constrained, ++, constrained++"

	if args.r == "y":
		try:
			int(args.dim)
		except ValueError :
			print("la dimension doit être un int : --dim 5")
			sys.exit()
		except TypeError:
			print("la dimension doit être un int : --dim 5")
			sys.exit()

	if len(args.traits) == 2:
		if args.fusion_method == "somme" or args.fusion_method == "moyenne":
			assert args.r == "y", "pour fusionner les traits syntaxiques et linéaires il faut réduire la taille des vecteurs : --r y"
			assert int(args.dim) <= 5, "la taille des traits syntaxiques et linéaires doit être inférieure ou égale à 5 : --r y --dim 5"



	##########################


	with open(conll) as file:
		file_conll = file.read()
	with open(gold) as file2:
		file_gold = file2.readlines()
	with open(tok_id) as file3:
		file_ids = file3.readlines()



	vectors_syntx,num_senses,vectors_linear,phrases=read_conll(file_conll, file_gold, file_ids, args.n,inventaire,args.tfidf, args.linear_method)

	# On ne prend que 5 premiers traits des vecteurs syntaxiques car marchent le mieux
	vectors_syntx = vectors_syntx[:, :5]

	# pondération tf-idf
	# if args.tfidf.lower() == "y" :
	# 	vectors_linear = tfidf(phrases, args.linear_method)

	if args.traits.lower() =='syntx':
		print("Only syntx traits")
		if args.r.lower()=='y': #si la reduction est demandé
			#np.savetxt("{}.txt".format(folder+'/'+argo.verbe+"_vectors_syntx"), vectors_syntx, delimiter = "\t")
			vectors_syntx=reduce_dimension(vectors_syntx,'syntx',argo.verbe,int(args.dim))
		else: 
			vectors_syntx=vectors_syntx
			
		examples=Examples()
		for i in range(len(vectors_syntx)):
			gold=file_gold[i].strip('\n')
			vector=Ovector(i,gold,None,vectors_syntx[i],None)
			vector.set_vector(vectors_syntx[i])
			examples.set_vector_to_matrix(vector)
				


	elif args.traits.lower() =='linear':
		print("Only linear traits")
		if args.r.lower()=='y':
			#np.savetxt("{}.txt".format(folder+'/'+argo.verbe+"_linear_vectors"), vectors_linear, delimiter = "\t")
			vectors_linear=reduce_dimension(vectors_linear,'linear',argo.verbe,int(args.dim))
		else:
			vectors_linear=vectors_linear

	
		examples=Examples()
		for i in range(len(vectors_linear)):
			vector=vectors_linear[i]
			gold=file_gold[i].strip('\n')
			vector=Ovector(i,gold,None,None,vectors_linear[i])
			vector.set_vector(vectors_linear[i])
			examples.set_vector_to_matrix(vector)


	elif args.traits.lower() =='deux':#si on demande tous les deux traits linear et syntx
		print('Linear et syntaxiques')
		if args.r.lower()=='y': #si la reduction est demandé
			#np.savetxt("{}.txt".format(folder+'/'+argo.verbe+"_vectors_syntx"), vectors_syntx, delimiter = "\t")
			#np.savetxt("{}.txt".format(folder+'/'+argo.verbe+"_linear_vectors"), vectors_linear, delimiter = "\t")
			vectors_linear=reduce_dimension(vectors_linear,'linear',argo.verbe,int(args.dim))
			vectors_syntx=reduce_dimension(vectors_syntx,'syntx',argo.verbe,int(args.dim))
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


	E = 200# nombre d'époques pour tourner l'algo



	matrix=examples.get_espace_vec()

	rd.shuffle(matrix)
	senses = Counter(int(line) for line in file_gold)
	#print("SENSES ", senses)
	N = len(senses.keys()) # le nb de clusters souhaité
	GOLD = senses.keys() # les numéros des sens, les classes gold
	#print(N)

	data_points=[]
	gold_points=[]
	len_vects=len(matrix[1].vector)
	for i in matrix:
		data_points.append(i.vector)
		gold_points.append(i.gold)

	"""df = pd.DataFrame.from_records(data_points,columns=['x','y'])
	df['labels']=gold_points
	print(df)
	"""
	#plot data with seaborn

	###INITIALIZING CENTERS+CLUSTERS WITH Kmeans++ ####
	if argo.cluster_type.lower() =='kmeans++' :
		centers=[]
		i=rd.randint(0,len(matrix))
		#print("I = ", i)
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

		#print("CENTERS")
		#print([(c.gold,c.vector) for c in centers])
		classification1 = KMeans(matrix, N, GOLD, None, args.dist_formula ,centers)
		classification1.create_empty_clustersPlus('n') #KMEANS ++



	### INITIALIZING CENTERS+CLUSTERS WITH CONSTRAINED++ ######
	if argo.cluster_type.lower() =='constrained++':
		centers=[]
		centers_gold=[]
		i=rd.randint(0,len(matrix))
		#print("I = ", i)
		center= examples.get_Ovector_by_id(i) #on rajoute une premier vector aléatoire comme centre
		centers_gold.append(examples.get_Ovector_by_id(i).gold)
		#print("FIRST CENTER = ", center.vector)
		centers.append(center)
		for cluster in range(N-1): #on va initialiser le k-1 autre centre de cluster qui reste
			distance=np.array([])
			for x in matrix: #chaque exemple dans l'espace
				point=x.vector
				#print("POINT LEN", point.shape)
				#print("CENTER ", center.vector.shape)
				distance=np.append(distance,np.min(np.sum((point-center.vector)**2)))
			
			proba=distance/np.sum(distance)
			cumul_proba=np.cumsum(proba)
			r=rd.random()
			i=0
			for j,p in enumerate(cumul_proba):
				if r<p: 
					if examples.get_Ovector_by_id(j).gold not in centers_gold:
						i=j
						break
			centers.append(examples.get_Ovector_by_id(j))
			centers_gold.append(examples.get_Ovector_by_id(j).gold)
			#print(centers_gold)
		#print("CENTERS")
		#print([(c.gold,c.vector) for c in centers])

		classification1 = KMeans(matrix, N, GOLD, None, args.dist_formula ,centers)
		classification1.create_empty_clustersPlus('y') #KMEANS ++
		

	#######VISUALISE CENTERS ################
	"""center_vectors=[c.vector for c in centers]
	center_pd=pd.DataFrame.from_records(center_vectors,columns=['x','y'])
	center_pd['golds']=centers_gold
				
	facet = sns.lmplot(data=df, x='x', y='y', hue='labels', fit_reg=False, legend=True, legend_out=True)
	#centers_points= sns.lmplot(data=center_pd, x='x', y='y', hue='golds', fit_reg=False, legend=True, legend_out=True,scatter_kws={"s": 100})
				
	plt.show()"""


	#####INITIALISATION OF CLUSTERS FOR CONSTRAINED ##########
	if argo.cluster_type.lower() == 'constrained':
		classification1 = KMeans(matrix, N, GOLD, None, args.dist_formula, None) # on n'a pas encore de centres
		classification1.create_empty_clusters() # constrainted Kmeans

	######################## CLUSTERING ############################

	############ KMEANS++ ###########

	if argo.cluster_type.lower() == "kmeans++":
		print("KMEANS++")
		c_centers1=[]
		for i in classification1.centers:
			c_centers1.append(i.vector) #list de centres
		tour1=0
		
		while True:
			for cluster_id in classification1.clusters:
				classification1.clusters[cluster_id].delete_examples()
				#classification1.clusters[cluster_id].resave_initial_example()
			if tour1 != E:
				tour1+=1
				for cluster_id in classification1.clusters:
					classification1.clusters[cluster_id].delete_examples()
					#classification1.clusters[cluster_id].resave_initial_example()
				for exo in classification1.examples:
					distances = []
					for cluster_id in classification1.clusters:
						#print("INITIAL : ", classification1.clusters[cluster_id].initial_example)
						if exo != classification1.clusters[cluster_id].initial_example:
							if args.dist_formula.lower() == 'cosine':
								distances.append(cosine(exo.vector, classification1.clusters[cluster_id].center))
							if args.dist_formula.lower() == 'euclidean':
								distances.append(euclidean(exo.vector, classification1.clusters[cluster_id].center))
							if args.dist_formula.lower() == 'cityblock':
								distances.append(cityblock(exo.vector, classification1.clusters[cluster_id].center))
					minimum_distance = np.argmin(distances)
					classification1.clusters[minimum_distance].add_example_to_cluster(exo)
				new_centers1=[]
				for cluster in classification1.clusters:
					classification1.clusters[cluster].recalculate_center()
					new_centers1.append(classification1.clusters[cluster].center)
				#####SI LES CENTRES CHANGE OU PAS########
				count1=0
				for i in range(len(c_centers1)):
					if np.all(c_centers1[i]==new_centers1[i]):
						count1+=1
				if count1 == len(c_centers1):		
					break
				else:
					c_centers1 = new_centers1
					new_centers1=[]

		print("RESULTS : ")
		list_exemples=[]
		list_golds=[]	
		cluster_dict1={}
		for i in classification1.clusters:
			classif1=Counter([exo.gold for exo in classification1.clusters[i].examples if exo !=None])
			classification1.clusters[i].redefine_id(max(classif1,key=classif1.get)) #id de cluster == la classe le plus nombreaux
			cluster_dict1["Cluster"+str(i)+"_gold: "+str(classification1.clusters[i].id)]=classif1
			#print("CLUSTER ", i)
			#print(len(classification1.clusters[i].examples))
			#print(classification1.clusters[i].id)
			#print(classif1)
			#print('\t')
		print(cluster_dict1)

		eval1=evaluate2(classification1.clusters)
		cluster_dict1["Fscore"]=eval1
		print("Fscore: ",eval1)

	###########CONSTRAINED ET CONSTRAINED++ #######################

	if argo.cluster_type.lower() == "constrained" or "constrained++":
		print("CONSTRAINED")

		c_centers1=[]
		for i in classification1.centers:
			c_centers1.append(i.vector) #list de centres
		tour1=0

		while True:
			for cluster_id in classification1.clusters:
				classification1.clusters[cluster_id].delete_examples()
				classification1.clusters[cluster_id].resave_initial_example()
			if tour1 != E:
				for cluster_id in classification1.clusters:
					classification1.clusters[cluster_id].delete_examples()
					classification1.clusters[cluster_id].resave_initial_example()
				distance = classification1.distance_matrix(args.dist_formula)
				for j in range(len(distance.T)): # on parcourt les exemples ; il faut savoir à quel id des exemples correspond j
					min_value_index = np.argmin(distance.T[j]) # on trouve l'indice de la valeur min; c'est l'id du CLUSTER
					exo = matrix[j] # exo à ajouter dans le cluster #min_value_index
					if exo != classification1.clusters[min_value_index].initial_example :
						classification1.clusters[min_value_index].add_example_to_cluster(exo) # j = quel id de l'exemple ?
				new_centers1=[]
				for cluster in classification1.clusters:
					classification1.clusters[cluster].recalculate_center()

					new_centers1.append(classification1.clusters[cluster].center)

					
				#####SI LES CENTRES CHANGE OU PAS########
				count1=0
				for i in range(len(c_centers1)):
					if np.all(c_centers1[i]==new_centers1[i]):
						count1+=1
				if count1 == len(c_centers1):		
					break
				else:
					c_centers1 = new_centers1
					new_centers1=[]
				tour1+=1

			else:
				break

		print("RESULTS:")
		cluster_dict1={}
		for i in classification1.clusters:
			classif1=Counter([exo.gold for exo in classification1.clusters[i].examples if exo !=None])
			classification1.clusters[i].redefine_id(max(classif1,key=classif1.get)) #id de cluster == la classe le plus nombreaux
			cluster_dict1["Cluster"+str(i)+"_gold: "+str(classification1.clusters[i].id)]=classif1
			#print("CLUSTER ", i)
			#print(len(classification1.clusters[i].examples))
			#print(classification1.clusters[i].id)
			#print(classif1)
			#print('\t')
		print(cluster_dict1)

		eval1=evaluate2(classification1.clusters)
		
		cluster_dict1["Fscore"]=eval1
		print("Fscore: ",eval1)
		
		result_dict=vars(args)
		result_dict.update({'Fscore':eval1})
		print("Resultat enregistré au flux")
		all_results.append(result_dict)

	#####CREER FICHIER POUR ECRITURE DES RESULTATS#######
	print("Ecriture des résultats")
	folder="Results_"+argo.verbe+"_"+argo.cluster_type
	os.makedirs(folder, exist_ok=True) #on cree un dossier pour chque essaie afin d'enregistrer les essaies
	outfile="{}.csv".format(folder+'/'+ "ARGS_FSCORE_RESULTS") #ECRITURE DES RESULTATS
	field_names=all_results[0].keys()
	with open(outfile, 'w', newline='') as csvfile:
		w = csv.DictWriter(csvfile,fieldnames=field_names)
		w.writeheader()
		w.writerows(all_results)

