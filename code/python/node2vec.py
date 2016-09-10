import numpy as np
import random
from scipy.special import expit
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import MeanShift
from collections import defaultdict
import time
import cPickle as pickle
from zutils.formula import listCosine

def normed(x):
	return x/np.linalg.norm(x,ord=1)

class Node2vec:
	def __init__(self):
		self.vec = None
		self.start_time = time.time()

	def train(self, net_u, net_v, dim=100, rand_seed=7, alpha=0.025, min_alpha=0.0001,
		epochs=1000, nagative=5):
	
		np.random.seed(rand_seed)
		random.seed(rand_seed)
	
		nodes_u, nodes_v = net_u.keys(), net_v.keys()
		self.vec = {node:np.random.rand(dim) for node in nodes_u+nodes_v}
		u2d = {u:len(net_u[u]) for u in net_u}
		u2samples = {u:iter(np.random.choice(net_u[u].keys(), u2d[u]*epochs*2, 
							p=normed(net_u[u].values()) )) for u in net_u}
		sample_size = epochs*sum(u2d.values())
		print sample_size

		for i,u in enumerate(np.random.choice( u2d.keys(), sample_size, p=normed(np.power(u2d.values(),4.0/3)) )):
			alpha -= (alpha - min_alpha) / sample_size
			try:
				v = u2samples[u].next()
			except StopIteration:
				u2samples[u] = iter(np.random.choice(net_u[u].keys(), u2d[u]*epochs*2, p=normed(net_u[u].values()) ))
				v = u2samples[u].next()
			for j in range(nagative+1):
				if j==0:
					target = v
					label = 1
				else:
					target = np.random.choice(nodes_v)
					if target in net_u[u]: continue
					label = 0
				f = np.dot(self.vec[u],self.vec[target])
				g = (label - expit(f)) * alpha
				self.vec[u] += g*self.vec[target]
				self.vec[target] += g*self.vec[u]
				# print u,j,target,g
			if i%10000==0:
				print i, time.time()-self.start_time

	def similarity(self, n1, n2):
		return listCosine(self.vec[n1],self.vec[n2])

def train_20news():
	twenty_test = fetch_20newsgroups(subset='test', random_state=42, categories=['rec.sport.baseball'])
	# X = twenty_test.data
	X = ['river water','river water table','river water','table cup','What the big deal about long games']
	cv = CountVectorizer()
	X = cv.fit_transform(X)
	X = TfidfTransformer().fit_transform(X)
	index2word = cv.get_feature_names()
	net_u, net_v = defaultdict(dict), defaultdict(dict)
	X = X.tocoo()
	for row,col,weight in zip(X.row, X.col, X.data):
		row = str(row)+' '
		col = index2word[col]
		net_u[row][col] = weight
		net_v[col][row] = weight
	node2vec = Node2vec()
	node2vec.train(net_u,net_v)
	pickle.dump(node2vec,open('node2vec.model','w'))
	print node2vec.similarity('river','water')
	print node2vec.similarity('river','table')
	print node2vec.similarity('river','cup')
	print node2vec.similarity('river','big')
	print node2vec.similarity('river','1 ')
	print node2vec.similarity('river','4 ')

def test_20news():
	node2vec = pickle.load(open('node2vec.model','r'))
	print node2vec.similarity('baseball','game')
	print node2vec.similarity('baseball','person')
	print node2vec.similarity('baseball','series')
	print node2vec.similarity('baseball','table')
	print node2vec.similarity('baseball','watch')


if __name__ == '__main__':
	train_20news()