#creation des vecteurs de contexte pour les fichiers conll
import numpy as np
import pandas as pd
from io import StringIO
from collections import defaultdict

def create_linear_ids(lines_phrase,lemme,tok_ids,n):
	keptwords=[l.split('\t')[2] for l in lines_phrase if l.split('\t')[3] in ['N','P','V','A','ADV']]
	#print('KEPTWORDS:',keptwords)
	nb_nul_debut = 0
	nb_nul_fin = 0
	verb_index=keptwords.index(lemme)
	#print('INDEX:',verb_index)
	if verb_index-1-n < 0:
		nb_nul_debut = n-(verb_index-1)
	if verb_index+n>=len(keptwords):
		nb_nul_fin = n-(len(keptwords)-(verb_index))
	# on crée des linears de contexte
	# on inclue le mot NULL au début et à la fin le nb de fois nécessaire

	linear = ["BOS" for k in range(nb_nul_debut)]
	linear.extend([l for l in keptwords[verb_index-1-n+nb_nul_debut:verb_index+n-nb_nul_fin] if l != keptwords[verb_index]])
	linear.extend(["EOS" for k in range(nb_nul_fin)])

	return linear

def get_linear_vectors():
	dict_word_vec=defaultdict()
	with open("vecs100-linear-frwiki.txt", "r") as fichier:
		next(fichier) #saute le premier ligne
		for i in fichier:
			mot=i.split(' ')
			dict_word_vec[mot[0].lower()]=np.asarray(mot[1:-1]).astype(np.float)
	return dict_word_vec