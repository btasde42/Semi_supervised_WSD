import argparse
import os
from collections import Counter
import numpy as np
import xlrd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from word_embed import create_ngram_ids, calcul_wordembeds, get_vectors_by_keys

def read_conll (conll, gold, tok_ids, n):
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
	vocabulary = Counter() #vocabulaire { mot : nb d'occurrence }
	list_keys=[]
	#mot_nul = "nul"
	ngrams = [] # # liste des ngrams
	#ngrammes = n mots (le nb de mots de contexte est passé en paramètre) avant et après le verbe ambigu (le verbe lui-même n'y est pas inclu)
	conll = conll.split('\n\n')[:-1]
	assert len(conll) == len(tok_ids)
	for i in range(len(tok_ids)):
		dependents = []
		verb_index = int(tok_ids[i])
		lines_phrase = conll[i].split('\n')
		lemme = lines_phrase[verb_index-1].split('\t')[2] # abattre lemme
		list_keys.append(lemme+'_'+str(i)) #on met les nombres sur chaque lemme abattre de type abattre_1...abattre_160
		ngram=create_ngram_ids(lines_phrase,verb_index,tok_ids,n)
		ngrams.append(ngram)
		for j,line in enumerate(lines_phrase):
			vocabulary[line.split('\t')[2]]+=1
			vocabulary[lemme+'_'+str(i)]+=1
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
	#vecteurs =np.zeros((len(tok_ids), 19), float) # à vérifier la taille des vecteurs
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

	ngrams_model=calcul_wordembeds(ngrams,lemme+'_'+str(i),list(vocabulary.keys()),5,19,n)
	ngram_vectors= np.empty((0, 19), float)
	for i in list_keys:
		vec=get_vectors_by_keys(ngrams_model,vocabulary,i)
		np_vec=vec.detach().numpy() #transform tensors to np arrays
		ngram_vectors=np.vstack((ngram_vectors,np_vec))
	np.savetxt(args.verbe+"_ngram_vectors", ngram_vectors, delimiter = "\t")
	return vectors, sujet, objet, ngram_vectors

def read_inventaire (inventaire, vectors, sujet, objet):
	num_senses = 0
	wb = xlrd.open_workbook(args.inventaire)
	sheet = wb.sheet_by_index(0)
	sheet.cell_value(0, 0)
	senses_suj = {}
	senses_obj = {}
	last_index = 0
	for i in range(sheet.nrows) :
		if sheet.row_values(i)[0] == args.verbe:
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
				vectors[key][6+num_senses+1] = 1 
	np.savetxt(args.verbe+"_vectors", vectors, delimiter = "\t")
	return vectors

def reduce_dimension(vectors,ngram_vectors):
	# Réduction de dimentionalité
	# noramlisation
	sc = StandardScaler() 
	vectors = sc.fit_transform(vectors)
	pca = decomposition.PCA(n_components = 11)
	vectors = pca.fit_transform(vectors)
	np.savetxt(args.verbe+"_reduced_vectors", vectors, delimiter = "\t")


	# variance exmpliquée : pour voir l'importance de chaque composante
	# combien d'information est gardé quand on réduit la taille des vecteurs
	explained_variance = pca.explained_variance_ratio_
	for elt in explained_variance:
		print(round(elt,2))
	return vectors


parser = argparse.ArgumentParser()
parser.add_argument("verbe", help = "abattre, aborder, affecter, comprendre, compter")
parser.add_argument('conll', help = 'le fichier conll full path')
parser.add_argument('gold', help = 'le fichier classe gold full path')
parser.add_argument('tok_ids', help = 'fichier tokens ids full path')
parser.add_argument("inventaire", help = 'inventaire de sens full path')
args = parser.parse_args()

verbes = ["abattre", "aborder", "affecter", "comprendre", "compter"]
# on s'assure que les fichiers correspondent au verbe sélectionné
assert args.verbe in verbes
assert args.verbe in args.conll 
assert args.verbe in args.gold
assert args.verbe in args.tok_ids


#os.chdir("./data/" + args.verbe + "/")

with open(args.conll) as file:
	file_conll = file.read()
with open(args.gold) as file2:
	file_gold = file2.readlines()
with open(args.tok_ids) as file3:
	file_ids = file3.readlines()

vectors, sujet, objet, ngram_vectors = read_conll(file_conll, file_gold, file_ids, 3)
new_vectors = read_inventaire(args.inventaire, vectors, sujet, objet)
reduced_vectors = reduce_dimension(new_vectors,ngram_vectors)
