import numpy as np
import random
from sklearn.cluster import MeanShift
from collections import defaultdict
import time, datetime
import cPickle as pickle
from zutils.formula import listCosine
import itertools
import sys
from io_utils import IO
from subprocess import call

def normed(x):
	return x/np.linalg.norm(x,ord=1)

def K(u,h=1):
	u /= h
	return ( (math.e**(-u*u/2))/math.sqrt(2*math.pi) ) / h

class Gsm2vecPredictor:
	def __init__(self,pd):
		self.pd = pd
		self.lClus = pd["lClus"](pd["bandwidth_l"])
		self.tClus = pd["tClus"](pd["bandwidth_t"])
		self.gsm2vec = Gsm2vec_line(pd)

	def fit(self, tweets):
		# print pd
		nt2nodes,et2net = self.prepare_training_data(tweets)
		self.gsm2vec.fit(nt2nodes,et2net)

	def prepare_training_data(self, tweets):
		nt2nodes = {nt:set() for nt in self.pd["ntList"]}
		et2net = {et:defaultdict(lambda : defaultdict(int)) for et in self.pd["etList"]}
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat,tweet.lng] for tweet in tweets]
		times  = [[self.pd["convert_ts"](tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)

		# encode_continuous_proximity
		print "encoding_continuous_proximity"
		self.encode_continuous_proximity('ll',self.lClus,et2net)
		self.encode_continuous_proximity('tt',self.tClus,et2net)
		print "encoded_continuous_proximity"

		# load voca
		word_localness = pickle.load(open(IO().models_dir+'word_localness.model','r'))
		voca = set(zip(*word_localness)[0])
		voca.remove("")

		for location,time,text,l,t in zip(locations,times,texts,ls,ts):
			nt2nodes['l'].add(l)
			nt2nodes['t'].add(t)
			et2net['lt'][l][t] += 1
			et2net['tl'][t][l] += 1
			# topl2weight = self.lClus.tops(location)
			# topt2weight = self.tClus.tops(time)
			# for topt in topt2weight:
			# 	et2net['lt'][l][topt] += topt2weight[topt]
			# for topl in topl2weight:
			# 	et2net['tl'][t][topl] += topl2weight[topl]
			words = [w for w in text if w in voca] # from text, only retain those words appearing in voca
			for w in words:
				nt2nodes['w'].add(w)
				et2net['tw'][t][w] += 1
				et2net['wt'][w][t] += 1
				et2net['wl'][w][l] += 1
				et2net['lw'][l][w] += 1
				# for topt in topt2weight:
				# 	et2net['wt'][w][topt] += topt2weight[topt]
				# for topl in topl2weight:
				# 	et2net['wl'][w][topl] += topl2weight[topl]
			for w1,w2 in itertools.combinations(words,r=2):
				if 'ww' in et2net:
					et2net['ww'][w1][w2] += 1
					et2net['ww'][w2][w1] += 1
		return nt2nodes, et2net

	def encode_continuous_proximity(self,et,clus,et2net):
		if et in et2net:
			centers = clus.get_centers()
			nodes = range(len(centers))
			edges = []
			i = 0
			for n1,n2 in itertools.combinations(nodes,r=2):
				proximity = np.linalg.norm(np.array(centers[n1])-np.array(centers[n2]),ord=2)**(-2)
				edges.append((str(n1),str(n2),proximity))
			edges.sort(key=lambda tup:tup[2],reverse=True)
			for n1,n2,proximity in edges[:len(nodes)*self.pd["nb_num"]]:
				et2net[et][n1][n2] = proximity

	def predict(self,time,lat,lng,words):
		nt2vecs = self.gsm2vec.nt2vecs
		l = self.lClus.predict([lat,lng])
		t = self.tClus.predict([self.pd["convert_ts"](time)])
		l_vec = nt2vecs['l'][l] if l in nt2vecs['l'] else np.zeros(self.pd["dim"])
		t_vec = nt2vecs['t'][t] if t in nt2vecs['t'] else np.zeros(self.pd["dim"])
		w_vecs = [nt2vecs['w'][w] for w in words if w in nt2vecs['w']]
		words_vec = np.average(w_vecs,axis=0) if w_vecs else np.zeros(self.pd["dim"])
		score = listCosine(l_vec,t_vec)+listCosine(t_vec,words_vec)+listCosine(words_vec,l_vec)
		return round(score,6)

	def get_vec(self,query):
		nt2vecs = self.gsm2vec.nt2vecs
		if isinstance(query,str):
			return nt2vecs['w'][query]
		elif isinstance(query,list):
			return nt2vecs['l'][self.lClus.predict(query)]
		else:
			return nt2vecs['t'][self.tClus.predict([query])]

	def get_nbs1(self,query,nb_nt,neighbor_num=20):
		vec_query = self.get_vec(query)
		candidates = [(nb,listCosine(vec_query,vec_nb)) for nb,vec_nb in self.gsm2vec.nt2vecs[nb_nt].items()]
		candidates.sort(key=lambda tup:tup[1],reverse=True)
		return candidates[:neighbor_num]

	def get_nbs2(self,query1,query2,func,nb_nt,neighbor_num=20):
		vec_query1 = self.get_vec(query1)
		vec_query2 = self.get_vec(query2)
		candidates = [(nb,func(listCosine(vec_query1,vec_nb),listCosine(vec_query2,vec_nb))) \
			for nb,vec_nb in self.gsm2vec.nt2vecs[nb_nt].items()]
		candidates.sort(key=lambda tup:tup[1],reverse=True)
		return candidates[:neighbor_num]

	def convert_freq_to_pmi(self):
		pass

class MeanshiftClus:
	def __init__(self,bandwidth):
		self.ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=5)

	def fit(self,X):
		X = np.array(X)
		self.ms.fit(X)
		return [str(label) for label in self.ms.labels_]

	def predict(self,x):
		return str(self.ms.predict([x])[0])

	def get_centers(self):
		return [list(center) for center in self.ms.cluster_centers_]

	# def tops(self,x):
	# 	candidates = [(clus,) for clus in range(len(self.ms.cluster_centers_))]


class LGridClus:
	def __init__(self,grid_num):
		self.grid_num = grid_num

	def fit(self,locations):
		lats,lngs = zip(*locations)
		self.min_lat, self.min_lng = min(lats),min(lngs)
		self.max_lat, self.max_lng = max(lats),max(lngs)
		return [self.predict(location) for location in locations]

	def predict(self,location):
		lat,lng = location
		lat = min(max(self.min_lat,lat),self.max_lat)
		lng = min(max(self.min_lng,lng),self.max_lng)
		lat_grid_len = (self.max_lat-self.min_lat)/self.grid_num
		lng_grid_len = (self.max_lng-self.min_lng)/self.grid_num
		i = int((lat-self.min_lat)/lat_grid_len)
		j = int((lng-self.min_lng)/lng_grid_len)
		return str(i*self.grid_num+j)

	def get_centers(self):
		lat_grid_len = (self.max_lat-self.min_lat)/self.grid_num
		lng_grid_len = (self.max_lng-self.min_lng)/self.grid_num
		return [ [self.min_lat+i*lat_grid_len, self.min_lng+j*lng_grid_len] \
			for i in range(self.grid_num) for j in range(self.grid_num)]

class TGridClus:
	def __init__(self,grid_num):
		self.grid_num = grid_num

	def fit(self,times):
		self.min, self.max = min([time[0] for time in times]),max([time[0] for time in times])
		return [self.predict(time) for time in times]

	def predict(self,time):
		time = time[0]
		time = min(max(self.min,time),self.max)
		grid_len = (self.max-self.min)/self.grid_num
		return str(int((time-self.min)/grid_len))

	def get_centers(self):
		grid_len = (self.max-self.min)/self.grid_num
		return [[self.min+i*grid_len] for i in range(self.grid_num)]

class TVoidClus:
	def __init__(self,place_holder):
		pass

	def fit(self,times):
		return [self.predict(time) for time in times]

	def predict(self,time):
		return str(time[0])

	def get_centers(self):
		return [ [i] for i in range(24) ]

class Gsm2vec_line:
	def __init__(self,pd):
		self.pd = pd
		self.nt2vecs = dict()
		self.io = IO()

	def fit(self, nt2nodes, et2net):
		self.write_line_input(nt2nodes, et2net)
		self.execute_line()
		self.read_line_output()

	def write_line_input(self, nt2nodes, et2net):
		for nt,nodes in nt2nodes.items():
			print nt,len(nodes)
			node_file = open(self.io.line_dir+"node-"+nt+".txt",'w')
			for node in nodes:
				node_file.write(node+"\n")
		for et,net in et2net.items():
			edge_file = open(self.io.line_dir+"edge-"+et+".txt",'w')
			for u,u_nb in net.items():
				for v,weight in u_nb.items():
					edge_file.write("\t".join([u,v,str(weight),"e"])+"\n")

	def execute_line(self):
		command = ["./hin2vec"]
		command += ["-size", str(self.pd["dim"])]
		command += ["-negative", str(self.pd["negative"])]
		command += ["-alpha", str(self.pd["alpha"])]
		command += ["-samples", str(self.pd["samples"])]
		command += ["-threads", str(self.pd["threads"])]
		call(command, cwd=self.io.line_dir)

	def read_line_output(self):
		for nt in self.pd["ntList"]:
			if nt==self.pd["predict_type"]:
				vecs_file = open(self.io.line_dir+"context-"+nt+".txt",'r')
			else:
				vecs_file = open(self.io.line_dir+"output-"+nt+".txt",'r')
			vecs = dict()
			for line in vecs_file:
				node, vec_str = line.strip().split("\t")
				vecs[node] = np.array([float(i) for i in vec_str.split(" ")])
			self.nt2vecs[nt] = vecs

