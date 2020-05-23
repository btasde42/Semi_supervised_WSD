import sys
import numpy as np
from math import *
import scipy.spatial

class Exemple:
	def __init__(self, example_number, gold_class):
		self.vector=Ovector()
		self.classe_gold=gold_class #prise du fichier .gold
		self.indice=example_number #prise du fichier .tok_ids
		self.example_matrix=np.empty((0,3), int) #pas encore bien definie
	
	def add_to_matrix(self,indice,vector,example_matrix):
		self.example_matrix[:,self.indice]=self.vector #remplacer le vecteur de cet exemple dans matrix

class Ovector:
	
	def __init__(self,list_traits,methode):
		self.list_traits=list_traits
		self.methode=methode #methode de fusion des vecteurs

	def fusion_traits(self): #comment on va fusionner les traits pour créer un seul vecteur par exemple
		if self.methode == 'somme' :
			for i in self.traits:
				vector_traits=np.array(i)

		if self.methode == 'moyenne':
			for i in self.traits:
				vector_traits=np.array(i)

		if self.methode == 'concetanation' :
			for i in self.traits:
				vector_traits=np.concetenate(vector_traits,np.array(i))

		return vector_traits

class Cluster:
	""" Le class de l'objet cluster """
	def __init__(self,id_cluster,examples, center):
		"""Args:
			id: le id de cluster sois int sois str
			examples: les exemples appertenant à cluster
		"""
		self.id=id_cluster
		self.examples=examples  #les exemples associées à cette cluster de sens, type:matrix
		self.center = center # l'exemple centre
	def add_example_to_cluster(self,example):
		self.examples = np.append(self.examples, example)


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
		# quand on ajoute un exemple du cluster, on supprimer cet exo de l'ensemble des exemples
		for i in range(self.k):
			self.clusters[i] = Cluster(i, self.examples[i], self.examples[i])
			self.examples = np.delete(self.examples, i, axis=0)
		return self.clusters

	def distance_matrix(self, distance_formula):
		"""distance_formula : string, ex cosine, euclidean, cityblock (for manhattan distance)
		"""
		centers = [self.clusters[i].center for i in range(self.k)]
		# IL FAUT PRENDRE EN COMPTE LES EXOS QUI NE SONT PAS DANS LES CLUSTERS ! 
		return scipy.spatial.distance.cdist(centers, self.examples, distance_formula)
	
	def retun_final_clusters(self):
		pass
class Hierarchy:

	def __init__(self,examples,contraints,clustering_type, clusters):
		self.contraints=contraints
		self.examples=examples
		self.clustering_type=clustering_type
		if self.clustering_type == 'ascending': #si la methode est ascending on  a le nombre des exemples comme nombre des clusters
			self.nbr_cluster=len(self.exemples)
		if self.clustering_type == 'descanding': #si la methode est descanding, on commence par un seul grande cluster
			self.nbr_cluster=1
		self.clusters = clusters

	def update_nbr_clusters(self,n):
		"""Remettre à jour le nombre se clusters """
		self.nbr_cluster=n

	def initialize_clusters(self):
		"""
		Creer les objets de clusters dans le nombre qu'on a besoin
		"""
		#S
		#for ex in self.examples:
			#clusters.add
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

