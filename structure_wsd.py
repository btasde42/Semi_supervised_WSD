import sys
import numpy as np
from math import *

class Example:
	def __init__(self):
		self.espace_vec=[]

	def set_vector_to_matrix(self,ovec):
		self.espace_vec.append(ovec)

	def get_espace_vec(self):
		return self.espace_vec

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
			id: le id de cluster sois int sois str
			examples: les exemples appertenant à cluster
		"""
		self.id=id_cluster
		self.examples=examples #les exemples associées à cette cluster de sens, type:matrix

	def add_exemple_to_cluster(self,example):
		pass

class KMeans:

	def __init__(self, exemples, k,contraints,distance_formula):
		self.k=k
		self.conraints=contraints
		self.distance_formula=distance_formula
		self.examples=exemples #pas encore bien etablie

	def stock_clusters():
		pass

	def create_empty_clusters(self):
		pass

	def distance_matrix(self):
		pass
	
	def retun_final_clusters(self):
		pass

class Hierarchy:

	def __init__(self,exemples,contraints,clustering_type):
		self.contraints=contraints
		self.exemples=exemples
		self.clustering_type=clustering_type
		if self.clustering_type == 'ascending': #si la methode est ascending on  a le nombre des exemples comme nombre des clusters
			self.nbr_cluster=len(self.exemples)
		if self.clustering_type == 'descanding': #si la methode est descanding, on commence par un seul grande cluster
			self.nbr_cluster=1

	def stock_clusters():
		#liste ou à penser
		pass
	
	def update_nbr_clusters(self,n):
		"""Remettre à jour le nombre se clusters """
		self.nbr_cluster=n

	def initialize_clusters(self):
		"""
		Creer les objets de clusters dans le nombre qu'on a besoin
		"""
		pass

	def merge_clusters(self):
		"""Methode pour fusionner les clusters si le methode est ascending """
		pass

	def demerge_clusters(self):
		"""Methode pour defusionner les clusters si le méthode est descending """
		pass

	def distance_matrix(self):
		pass

	def retun_final_clusters(self):
		pass

class WSD:
	def __init__(self,method,examples):
		self.method = method
		self.exampls=examples
	
	def create_clustering(self,examples,k,contraints,distance_formula):
		if self.method == "kmeans":
			return KMeans(self.examples,k,contraints,distance_formula)
		if self.method=='hierarchy':
			return Hierarchy(self.examples,contraints,clustering_type)

	def evaluate(self,examples):
		pass

