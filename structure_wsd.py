import sys
import numpy as np
from math import *

class Examples:
	def __init__(self):
		self.espace_vec=[]

	def set_vector_to_matrix(self,ovec):
		self.espace_vec.append(ovec)

	def get_espace_vec(self):
		return self.espace_vec
	
	def get_id_from_vector(self,vector):
		"""
		Returns the id of given vector in the vector space
		"""
		for i in self.espace_vec:
			if i.get_vector()==vector:
				return i.get_index()
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
	def __init__(self,id_cluster,examples):
		"""Args:
			id: le id de cluster sois int sois str -> id correspond au id du sens
			examples: les exemples appertenant à cluster
		"""
		self.id=id_cluster
		#self.examples=examples  #les exemples associées à cette cluster de sens, type:matrix
		self.examples = []
		self.examples.append(examples)
		self.center = np.mean(self.examples, axis = 0)
	
	def add_example_to_cluster(self,example):
		#self.examples = np.append(self.examples, example)
		self.examples.append(example)
		# on fait la màj du centre -> c'est la moyenne des exemples
		#self.center = np.mean(self.examples, axis = 0)
	
	def recalculate_center(self):
		self.center = np.mean(self.examples, axis=0)
	
	def delete_examples(self):
		self.examples = []

class KMeans:

	def __init__(self, examples, k,contraints,distance_formula):
		self.k=k
		self.conraints=contraints
		self.distance_formula=distance_formula
		self.examples=examples #pas encore bien etablie
		self.clusters = {}
	def create_empty_clusters(self):
		""" on initialis les k cluster avec le centre et un exemple
		"""
		# avant d'appliquer cette méthode il faut faire shuffle des exemples
		for i in range(self.k):
			self.clusters[i] = Cluster(i, self.examples[i]) #, self.examples[i])
			#self.examples = np.delete(self.examples, i, axis=0)
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
		centers = [self.clusters[i].center for i in range(self.k)]
		# IL FAUT PRENDRE EN COMPTE LES EXOS QUI NE SONT PAS DANS LES CLUSTERS ! 
		return scipy.spatial.distance.cdist(centers, self.examples, distance_formula)
	
	def retun_final_clusters(self):
		return self.clusters


class WSD:
	def __init__(self,clusters):
		self.clusters=clusters
	
	def evaluate(self,examples):
		"""dict_gold={}
								for c in clusters:
									for e in c.examples:"""
		pass

