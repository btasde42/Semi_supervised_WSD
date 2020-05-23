import create_vectors
import structure_wsd

parser = argparse.ArgumentParser()
parser.add_argument("verbe", help = "abattre, aborder, affecter, comprendre, compter")
parser.add_argument('conll', help = 'le fichier conll full path')
parser.add_argument('gold', help = 'le fichier classe gold full path')
parser.add_argument('tok_ids', help = 'fichier tokens ids full path')
parser.add_argument("inventaire", help = 'inventaire de sens full path')
parser.add_argument("reduced",help='Y ou N selon si on veut reduire les vecteurs')
parser.add_argument("traits",nargs='+',help="List des traits qu'on veut utiliser. [Syntx,ngram]")
parser.add_argument("--n",help='la taille de contexte pour les ngrams. Optionelle.')

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

vectors_ngram,vectors_syntx=read_conll(file_conll, file_gold, args.inventaire, file_ids, args.n)

if reduced.lower=='y':
	vectors_ngram=reduce_dimension(ngram_vectors,'ngram')
	vectors_syntx=reduce_dimension(ngram_vectors,'ngram')


Examples=Exemple(args.gold,ngram_vectors,syntax_vectors)