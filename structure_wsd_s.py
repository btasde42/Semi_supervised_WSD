import sys
import numpy as np
from math import *
import scipy.spatial
from collections import defaultdict, Counter

class Examples:
	def __init__(self):
		self.espace_vec=[]

	def set_vector_to_matrix(self,ovec):
		self.espace_vec.append(ovec)

	def get_espace_vec(self):
		return self.espace_vec

	def get_Ovector_by_id(self, id):
		for example in self.espace_vec:
			if example.index==id:
				return example
		print("Warning : example with id = ", id, " not found")
		return None
	
	def get_Ovector_by_vector(self, vector):
		for example in self.espace_vec :
			if (example.vector == vector).all():
				return example
		print("Warning : example with vector = ", vector, " not found")
		return None


class Ovector:
	def __init__(self,index,gold,methode=None,traits_syntaxique=None,traits_ngram=None):
		self.traits_syntaxique=traits_syntaxique if traits_syntaxique is not None else np.zeros(traits_ngram.shape)#si initialize son valeur sinon np.zeros
		self.traits_ngram= traits_ngram if traits_ngram is not None else np.zeros(traits_syntaxique.shape)
		self.methode=methode if methode is not None else None #methode de fusion des vecteurs
		self.index=index
		self.gold=gold
		self.vector=[[]]

	def fusion_traits(self): #comment on va fusionner les traits pour créer un seul vecteur par exemple
		
		if self.methode.lower() == 'somme' :
			self.vector=np.add(self.traits_syntaxique,self.traits_ngram)

		if self.methode.lower() == 'moyenne':
			moyenne_syntx=np.mean(self.traits_syntaxique)
			moyenne_ngram=np.mean(self.traits_ngram)
			self.vector=np.array([moyenne_syntx,moyenne_ngram]) #on cree un vecteur de taille (2,)
		if self.methode.lower() == 'moyenne2' :
			#print(type(self.traits_syntaxique))
			#print(np.mean((self.traits_syntaxique, self.traits_ngram), axis = 0))
			self.vector = np.mean((self.traits_syntaxique, self.traits_ngram), axis = 0)
			#return np.mean(np.array([traits_syntaxique, traits_linear]),axis=0) # un vecteur moyen de taille réduite (après la réduction avec ACP)
		if self.methode.lower() == 'concat' :
			self.vector=np.concatenate((self.traits_syntaxique,self.traits_ngram),axis=None)

	def set_vector(self,vec):
		self.vector=vec

	def get_vector(self):
		return self.vector

	def get_gold_class(self):
		return self.gold

	def get_index(self):
		return self.index

class Cluster:
	""" Le class de l'objet cluster """
	def __init__(self,id_cluster, center, examples, initial_example) : # initial_example=None):
		"""Args:
			id: le id de cluster sois int sois str -> id correspond au id du sens
			examples: les exemples appertenant à cluster
		"""
		# AJOUTER ATTRIBUT GOLD
		self.id=id_cluster
		self.initial_example=initial_example
		#self.gold = gold
		#self.examples=examples  #les exemples associées à cette cluster de sens, type:matrix
		self.examples = []
		self.examples.append(examples)
		self.center = center
		#print("SELF CENTER ", self.center)
		#self.center = np.mean(self.examples, axis = 0)
	
	def add_example_to_cluster(self,example):
		#self.examples = np.append(self.examples, example)
		self.examples.append(example)
		# on fait la màj du centre -> c'est la moyenne des exemples
		#self.center = np.mean(self.examples, axis = 0)
	
	def recalculate_center(self):
		#for example in self.examples:
		#	print(type(example))
		#	print(example.vector)
		self.center = np.mean([ example.vector for example in self.examples], axis=0)

	
	def delete_examples(self):
		self.examples = []
	
	def resave_initial_example(self):
		self.examples.append(self.initial_example)

	def redefine_id(self,n_id):
		self.id=n_id

class KMeans:
	_ID = 0
	def __init__(self, examples, k, gold, contraints,distance_formula,centers):
		self.k=k
		self.gold = gold # les noms des clusters=classes gold
		self.conraints=contraints
		self.distance_formula=distance_formula
		self.examples=examples 
		self.clusters = {}
		if centers != None: 
			self.centers=centers #si les centres sont donnés
		else:
			self.centers=[] #sinon

	def create_empty_clustersPlus(self,ifconstrained):
		""" Initialisation des clusters selon les centres provient de KMeans++
		if constrained++ there is initial example
		else if only Kmeans++ there isn't initial example
		"""
		for i in self.centers:
			#self.clusters[self._ID]=Cluster(i.gold,i.vector,i) #pas de initial exemple
			if ifconstrained.lower()=='y':
				self.clusters[self._ID]=Cluster(i.gold[0],i.vector,i,i)
			else:
				self.clusters[self._ID]=Cluster(i.gold[0],i.vector,i,None)
			self._ID+=1
		return self.clusters


	def create_empty_clusters(self):
		""" on initialis les k cluster avec le centre et un exemple
		"""
		# avant d'appliquer cette méthode il faut faire shuffle des exemples
		print("GOLD = ", self.gold)
		centers=[]
		for g in self.gold:
			print(type(g))
			for i in range(len(self.examples)):
				#print(self.examples[i].gold)
				if int(self.examples[i].gold[0]) == g:
					#print("TRUE")
					self.clusters[self._ID] = Cluster(g, self.examples[i].vector, self.examples[i], self.examples[i]) # à changer ensuite gold -> int 
					self.centers.append(self.examples[i])
					#print("initial ", self.clusters[self._ID].initial_example)
					self._ID+=1
					#print(i)
					break
		#for i in range(len(self.clusters)):
		#	print((self.clusters[i].center.gold, self.clusters[i].center.vector))
			#print("initial ", self.clusters[i].initial_example)
			#self.clusters[i] = Cluster(i, self.examples[i]) #, self.examples[i])
		#	#self.examples = np.delete(self.examples, i, axis=0) # no need
		return self.clusters

	def delete_example(self, example): # PAS BESOIN 
		"""on supprime l'exemple passé en paramètre de l'ensemble des exemples non-classés
		peut être utilisé après l'ajout de cet exemple dans un des clusters
		"""
		id_example = np.where(self.examples == example)
		self.examples = np.delete(self.examples, id_example, axis=0)


	def distance_matrix(self, distance_formula):
		"""distance_formula : string, ex cosine, euclidean, cityblock (for manhattan distance)
		"""
		centers = [self.clusters[i].center for i in range(self._ID)]#range(1, len(self.gold)+1)]
		#print("LEN CENTER ")
		#for center in centers:
		#	print(len(center))
		# IL FAUT PRENDRE EN COMPTE LES EXOS QUI NE SONT PAS DANS LES CLUSTERS ! 
		# return scipy.spatial.distance.cdist(centers, self.examples, distance_formula)
		if self.distance_formula.lower() == 'cosine':
			return scipy.spatial.distance.cdist(centers, [example.vector for example in self.examples], distance_formula)
		elif self.distance_formula.lower() == 'euclidean':
			return scipy.spatial.distance.cdist(centers, [example.vector for example in self.examples], distance_formula)		
		elif self.distance_formula.lower() == 'cityblock':
			return scipy.spatial.distance.cdist(centers, [example.vector for example in self.examples], distance_formula)
		else:
			print("Distance formule non-trouvé !!!")
	def retun_final_clusters(self):
		pass

def evaluate(clusters):
	dict_scores=defaultdict(dict)
	classes=Counter()
	Fscore_global=0.0
	
	for c in clusters:
		for e in clusters[c].examples:
			classes[e.gold]+=1 #on calcule le nombre de gold pour chaque classes

	for c in clusters:	
		gold_cluster=clusters[c].id
		vrai=0
		for e in clusters[c].examples:
			if e.gold==gold_cluster:
				vrai+=1


		precision=vrai/len(clusters[c].examples)
		rappel= vrai/classes[gold_cluster]
		F= 2* ((precision*rappel) / (precision+rappel))

		weight= classes[gold_cluster]
		Fscore_global+=(weight*F)/sum(classes.values())

	return Fscore_global

def evaluate2(clusters):
	dict_scores=defaultdict(dict)
	classes=Counter()
	#Fscore_global=0.0
	precision = 0.0
	rappel = 0.0
	for c in clusters:
		for e in clusters[c].examples:
			classes[e.gold]+=1 #on calcule le nombre de gold pour chaque classes

	for c in clusters:	
		gold_cluster=clusters[c].id
		vrai=0
		for e in clusters[c].examples:
			if e.gold==gold_cluster:
				vrai+=1


		precision+=vrai/len(clusters[c].examples)
		rappel+= vrai/classes[gold_cluster]
	precision = precision/len(classes)
	print("precision = ", precision)
	rappel = rappel/len(classes)
	print("rappel = ", rappel)
	F= 2* ((precision*rappel) / (precision+rappel))

	#weight= classes[gold_cluster]
	#Fscore_global+=(weight*F)/sum(classes.values())

	return F