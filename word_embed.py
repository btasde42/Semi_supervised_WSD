#creation des vecteurs de contexte pour les fichiers conll
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def create_ngram_ids(lines_phrase,verb_index,tok_ids,n):
	
	nb_nul_debut = 0
	nb_nul_fin = 0
	if verb_index-1-n < 0:
		nb_nul_debut = n-(verb_index-1)
	if verb_index+n>=len(lines_phrase):
		nb_nul_fin = n-(len(lines_phrase)-(verb_index))
	# on crée des ngrams de contexte
	# on inclue le mot NULL au début et à la fin le nb de fois nécessaire
	ngram = ["BOS" for k in range(nb_nul_debut)]
	ngram.extend([l.split('\t')[2] for l in lines_phrase[verb_index-1-n+nb_nul_debut:verb_index+n-nb_nul_fin] if l != lines_phrase[verb_index-1]])
	ngram.extend(["EOS" for k in range(nb_nul_fin)])

	return ngram
class NGramLanguageModeler(nn.Module):
	def __init__(self, vocab_size, embedding_dim, context_size):
		super(NGramLanguageModeler, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size*2 * embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((1, -1))
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs

def calcul_wordembeds(ngrams,verbe,vocab,n_epoch, EMBEDDING_DIM,CONTEXT_SIZE):
	vocab.append('EOS')
	vocab.append('BOS')
	word_to_ix = {word: i for i, word in enumerate(vocab)}

	losses = []
	loss_function = nn.NLLLoss()

	model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
	optimizer = optim.SGD(model.parameters(), lr=0.001)


	for epoch in range(n_epoch):
		total_loss = 0
		for i in ngrams:
			target=verbe
			context=i
			context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

			model.zero_grad()
			log_probs = model(context_idxs) #lance le modèle
			loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			
		losses.append(total_loss)
	return model

def get_vectors_by_keys(model,vocab,key):
	return model.embeddings(autograd.Variable(torch.LongTensor([vocab[key]])))


###TESTS
#vocab=set(test_sentence)
#print(create_ngrams_ids(test_sentence,5,3))
#print(calcul_wordembeds(test_sentence,vocab,5,7,3))