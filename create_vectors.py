import argparse
import os
from collections import Counter
import numpy as np
import xlrd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from word_embed import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

def read_conll(conll, gold, tok_ids, n, inventaire, iftfidf,method=None):
	"""la fonction qui lit le fichier conll et renvoie la matrice dont chaque ligne représente une occurrence du verbe
	conll - corpus au format conll
	gold - fichier avec les classes gold
	tok_ids - fichier avec la position du verbe ambigu dans la phrase
	n (int) - le nombre de mots du contexte
	"""
	suj_animé = {}
	sujet = {}
	obj_animé = {}
	objet = {}
	v_active = {} # 1 - active, 2 - passive
	root = {} # 1 - oui, 0 - non
	fonction = {} # V, VINF, VPP, VPR
	dist_suj = {}
	dist_obj = {}
	list_keys=[]
	phrases = []
	linear_window = []
	dict_mots_vec=get_linear_vectors()
	linear_vectors = [] # # liste des linears
	#linearmes = n mots (le nb de mots de contexte est passé en paramètre) avant et après le verbe ambigu (le verbe lui-même n'y est pas inclu)
	conll = conll.split('\n\n')[:-1]
	assert len(conll) == len(tok_ids)
	for i in range(len(tok_ids)):
		dependents = []
		verb_index = int(tok_ids[i])
		lines_phrase = conll[i].split('\n')
		phrase =' '.join([mot.split('\t')[2].lower() for mot in lines_phrase])
		phrases.append(phrase)
		lemme = lines_phrase[verb_index-1].split('\t')[2] # abattre lemme
		list_keys.append(lemme+'_'+str(i)) #on met les nombres sur chaque lemme abattre de type abattre_1...abattre_160

		####linear POUR L'EXEMPLE i###
		linear=create_linear_ids(lines_phrase,lemme,tok_ids,n) #creer fenetre de n
		linear_window.append(linear)
		vector_n_i=[]
		for j in linear:
			if j.lower() in dict_mots_vec: #si le mot se trouve dans le fichier vecteurs
				vector_n_i.append(dict_mots_vec[j.lower()])
			else: #sinon
				vec_zeros = np.zeros(100, float) # à vérifier la taille des vecteurs
				vector_n_i.append(vec_zeros)
		if iftfidf.lower() != "y" and method != None: #si linear demandée
			if method.lower() == 'somme':
				vector_n_i=np.vstack(vector_n_i).sum(axis=0)
			if method.lower() == 'moyenne':
				vector_n_i=np.mean(vector_n_i, axis=0)
			if method.lower() == 'concat':
				vector_n_i=np.concatenate(vector_n_i)
	
		linear_vectors.append(vector_n_i)

		############
		for j,line in enumerate(lines_phrase):

			if str(verb_index) in line.split('\t')[6].split('|'):
				list_indices = line.split('\t')[6].split('|')
				verb_index_dep = line.split('\t')[6].split('|').index(str(verb_index))
				if line.split("\t")[7].split('|')[verb_index_dep][:3] == "suj":
					if line.split('\t')[5][-6:] =="anim=I":
						suj_animé[i] = 0					
					else:
						suj_animé[i] = 1
					sujet[i] = line.split("\t")[2]
					dist_suj[i] = verb_index - int(line.split('\t')[0])
				if line.split("\t")[7].split('|')[verb_index_dep][:3] == "obj":
					if line.split('\t')[5][-6:] =="anim=I":
						obj_animé[i] = 0					
					else:
						obj_animé[i] = 1
					objet[i] = line.split("\t")[2]
					dist_obj[i] = verb_index - int(line.split('\t')[0])
		if lines_phrase[verb_index-1].split('\t')[5].split('|')[0][5:] == "passif":
			v_active[i] = 0
		else:
			v_active[i] = 1
		if lines_phrase[verb_index-1].split('\t')[7] == "root":
			root[i] = 1
		if lines_phrase[verb_index-1].split('\t')[4] == "VPP": # Si fonction == V, on laisse 0
			fonction[i] = 1
		elif lines_phrase[verb_index-1].split('\t')[4] == "VINF":
			fonction[i] = 2
		elif lines_phrase[verb_index-1].split('\t')[4] == "VPR":
			fonction[i] = 3
	vectors = np.zeros((len(tok_ids), 19), float) # à vérifier la taille des vecteurs

	for i in range(len(tok_ids)):
		if i in suj_animé.keys():
			vectors[i][0] = suj_animé[i]
		if i in obj_animé.keys():
			vectors[i][1] = obj_animé[i]
		if i in v_active.keys():
			vectors[i][2] = v_active[i]
		if i in root.keys():
			vectors[i][3] = root[i]
		if i in fonction.keys():
			vectors[i][4] = fonction[i]
		if i in dist_suj.keys():
			vectors[i][5] = dist_suj[i]
		if i in dist_obj.keys():
			vectors[i][6] = dist_obj[i]
	
	vectors_syntx,num_senses=read_inventaire(inventaire, vectors, sujet, objet,lemme)
	if iftfidf.lower() == "y":
		linear_vectors = tfidf2(phrases, linear_window, linear_vectors, method)
	#*************************
	#print(phrases2[:3])
	#tfidf2(phrases, linear_window, linear_vectors, method)
	#*************************
	return vectors_syntx,num_senses, linear_vectors, phrases

def read_inventaire (inventaire, vectors, sujet, objet,lemme):
	num_senses = 0
	wb = xlrd.open_workbook(inventaire)
	sheet = wb.sheet_by_index(0)
	sheet.cell_value(0, 0)
	senses_suj = {}
	senses_obj = {}
	last_index = 0
	for i in range(sheet.nrows) :
		if sheet.row_values(i)[0] == lemme:
			senses_suj[num_senses] = sheet.row_values(i)[4].split('/')
			senses_obj[num_senses] = sheet.row_values(i)[7].split('/')
			num_senses+=1
	for key in sujet.keys():
		for sense in senses_suj.keys():
			if sujet[key] in senses_suj[sense]:
				vectors[key][6+sense+1] = 1 # on commence à remplir à partir du dernier elt rempli du vecteur
	for key in objet.keys():
		for sense in senses_obj.keys():
			if objet[key] in senses_obj[sense]:
				vectors[key][6+sense+1] = 1 
	#print("MAX LEN VECTOR : ", 6+sense+1)
	return vectors,num_senses

def reduce_dimension(vectors,name_type,verbe,dimention):
	"""Give a set of vectors and reduce them by PCA
	Args:
		vectors: np array
		name_type: change by the type of vectors
	"""
	# Réduction de dimentionalité
	# noramlisation
	sc = StandardScaler() 
	vectors = sc.fit_transform(vectors)
	pca = decomposition.PCA(n_components = dimention)
	vectors = pca.fit_transform(vectors)
	#np.savetxt(verbe+"_reduced_vectors"+name_type, vectors, delimiter = "\t")

	# variance exmpliquée : pour voir l'importance de chaque composante
	# combien d'information est gardé quand on réduit la taille des vecteurs
	explained_variance = pca.explained_variance_ratio_
	#for elt in explained_variance:
		#print(round(elt,2))
	
	#print("TRY REDUCED VECTOR : ", vectors[0])
	return vectors

def fusion_traits(traits_syntaxique,traits_linear,method): #comment on va fusionner les traits pour créer un seul vecteur par exemple
		print("FUSION")
		if method.lower() == 'somme' :
			return np.add(traits_syntaxique,traits_linear)

		if method.lower() == 'moyenne':
			moyenne_syntx=np.mean(traits_syntaxique)
			moyenne_linear=np.mean(traits_linear)
			return np.array([moyenne_syntx,moyenne_linear]) #on cree un vecteur de taille (2,)
		if method.lower() == 'moyenne2' :
			#print(type(traits_syntaxique))
			#print(np.mean(traits_syntaxique, traits_linear))
			return np.mean(np.array([traits_syntaxique, traits_linear]),axis=0) # un vecteur moyen de taille réduite (après la réduction avec ACP)
		if method.lower() == 'concat':
			return np.concatenate(traits_syntaxique,traits_linear)

def tfidf(phrases, linear_window, linear_vectors, method) :
	"""linear_window = la fenêtre des mots
	linear_vectors = les vecteurs de contexte
	"""
	vectorizer = TfidfVectorizer(input=phrases)
	vectors_tfidf = vectorizer.fit_transform(phrases)
	vectors_tfidf = vectors_tfidf.todense()
	vocabulary = vectorizer.get_feature_names()

	for i in range(len(linear_window)) :
		for j in range(len(linear_window[i])):
			mot = linear_window[i][j]
			if mot in vocabulary:
				idx = vocabulary.index(mot.lower())
				weight = vectors_tfidf[i,idx]
				#if mot == "avoir":
					#print("WEIGHT AVOIR ", weight)
				linear_vectors[i][j] = linear_vectors[i][j]*weight
			else :
				#print("mot ", mot, "n'est pas trouvé")
				linear_vectors[i][j] = np.zeros(100, float)
		if method != None :
			if method.lower() == 'somme':
				linear_vectors[i]=np.vstack(linear_vectors[i]).sum(axis=0)
			if method.lower() == 'moyenne':
				linear_vectors[i]=np.mean(linear_vectors[i], axis=0)
			if method.lower() == 'concat':
				linear_vectors[i]=np.concatenate(linear_vectors[i])
	return linear_vectors

def tfidf2 (phrases, linear_window, linear_vectors, method):
	count_words = defaultdict(int)
	for phrase in phrases:
		for word in phrase.split():
			#print(word)
			count_words[word]+=1
	count_doc_words = defaultdict(int)
	tfidf = defaultdict(int)
	for word in count_words:
		tf = 0.0
		idf = 0.0
		tf = count_words[word]/sum(count_words.values())
		for phrase in phrases:
			if word in phrase.split():
				count_doc_words[word]+=1
		idf = len(phrases)/count_doc_words[word]
		tfidf[word] = tf*idf
	for i in range(len(linear_window)) :
		for j in range(len(linear_window[i])):
			mot = linear_window[i][j]
			if mot in tfidf.keys():
				#idx = vocabulary.index(mot.lower())
				#weight = vectors_tfidf[i,idx]
				weight = tfidf[mot]
				#if mot == "avoir":
					#print("WEIGHT AVOIR ", weight)
				linear_vectors[i][j] = linear_vectors[i][j]*weight
			else :
				#print("mot ", mot, "n'est pas trouvé")
				linear_vectors[i][j] = np.zeros(100, float)
		if method != None :
			if method.lower() == 'somme':
				linear_vectors[i]=np.vstack(linear_vectors[i]).sum(axis=0)
			if method.lower() == 'moyenne':
				linear_vectors[i]=np.mean(linear_vectors[i], axis=0)
			if method.lower() == 'concat':
				linear_vectors[i]=np.concatenate(linear_vectors[i])
	return linear_vectors