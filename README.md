# Semi_supervised_WSD
 Code contenant notre projet TAL M1 : La désambiguisation de sens semi-supervisée.


## Pour le lancement du code:

	python3 main.py "nomdeverbe" 'chemin_conll' 'chemin_gold' 'chemin_tok_ids' 'chemin_inventaire' --r 'y/n' --traits (linear, syntax ou les deux) --n (int) --fusion_method (concat, somme, moyenne) --linear_method (concat, somme, moyenne) --dim (int)

* --r : Y ou N selon si on veut reduire les vecteurs, IMPORTANT: #si on applique pas la reduction, fusion_method doit etre moyenne ou concat
* --traits : List des traits qu'on veut utiliser. [syntx,linear]
* --n : a taille de contexte pour les linears. Optionelle.
* --fusion_method : La methode de fusion pour differents types des vecteurs de traits s'il y en a plusieurs / Optionnelle

* --linear_method : Methode pour fusionner les vecteurs des mots pour les vecteurs linéaires.
* --dim : La taille de dimention reduit pour les vecteurs de verbe

### EXAMPLES:

	python3 main.py "abattre" "/path_conll" "/path_gold" "/path_tok_ids" "/path_inventaire" --r n --traits syntx linear --n 2 --fusion_method concat --linear_method moyenne

Produit des vecteurs pour la verbe abattre. La reduction de taille n'est pas appliqué. Les traits syntaxiques et linéaires sont fusionnées avec la concetanation. La méthode de fusion pour les fenetres de mots (linéaires) est la moyenne.
	
	python3 main.py "abattre" "/path_conll" "/path_gold" "/path_tok_ids" "/path_inventaire" --r y --traits linear --n 2 --linear_method somme	--dim 11

Produit des vecteurs pour la verbe abattre. La reduction de taille est appliqué pour la taille 11. Seulement les traits linéaires sont utilisées. La méthode de fusion pour les fenetres de mots (linéaires) est la somme.

	python3 main.py "affecter" "/path_conll" "/path_gold" "/path_tok_ids" "/path_inventaire" --r y --traits syntx --dim 10

Produit des vecteurs pour la verbe affecter. La reduction de taille est appliqué pour la taille 10. Seulement les traits syntaxiques sont utilisées.
