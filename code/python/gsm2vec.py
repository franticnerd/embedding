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
from scipy.special import expit
import math


class Gsm2vecPredictor:
	def __init__(self, pd):
		self.pd = pd
		self.lClus = pd["lClus"](pd)
		self.tClus = pd["tClus"](pd)
		self.nt2vecs = None

	def fit(self, tweets):
		gsm2vec = self.pd["gsm2vec"](self.pd)
		if isinstance(gsm2vec, Gsm2vec_relation):
			nt2nodes, relations = self.prepare_training_data_for_Gsm2vec_relation(tweets)
			self.nt2vecs = gsm2vec.fit(nt2nodes, relations)
		else:
			nt2nodes, et2net = self.prepare_training_data(tweets)
			self.nt2vecs = gsm2vec.fit(nt2nodes, et2net)

	def prepare_training_data_for_Gsm2vec_relation(self, tweets):
		nt2nodes = {nt:set() for nt in self.pd["ntList"]}
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times  = [[self.pd["convert_ts"](tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)

		# load voca
		word_localness = pickle.load(open(IO().models_dir+'word_localness.model', 'r'))
		voca = set(zip(*word_localness)[0])
		voca.remove("")

		relations = []
		for location, time, text, l, t in zip(locations, times, texts, ls, ts):
			relation = defaultdict(lambda : defaultdict(float))
			# for l, weight in self.lClus.tops(location):
			# 	nt2nodes['l'].add(l)
			# 	relation['l'][l] = weight
			# for t, weight in self.tClus.tops(time):
			# 	nt2nodes['t'].add(t)
			# 	relation['t'][t] = weight
			nt2nodes['l'].add(l)
			relation['l'][l] = 1
			nt2nodes['t'].add(t)
			relation['t'][t] = 1
			words = [w for w in text if w in voca] # from text, only retain those words appearing in voca
			for w in words:
				nt2nodes['w'].add(w)
				relation['w'][w] = 1
			relations.append(relation)

		return nt2nodes, relations

	def prepare_training_data(self, tweets):
		nt2nodes = {nt:set() for nt in self.pd["ntList"]}
		all_et = [nt1+nt2 for nt1, nt2 in itertools.product(self.pd["ntList"], repeat=2)]
		et2net = {et:defaultdict(lambda : defaultdict(float)) for et in all_et}
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times  = [[self.pd["convert_ts"](tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)

		# load voca
		word_localness = pickle.load(open(IO().models_dir+'word_localness.model', 'r'))
		voca = set(zip(*word_localness)[0])
		voca.remove("")

		for location, time, text, l, t in zip(locations, times, texts, ls, ts):
			nt2nodes['l'].add(l)
			nt2nodes['t'].add(t)
			et2net['lt'][l][t] += 1
			et2net['tl'][t][l] += 1
			# topl_weights = self.lClus.tops(location, self.pd["kernel_candidate_num"])
			# topt_weights = self.tClus.tops(time, self.pd["kernel_candidate_num"])
			# for topt, weight in topt_weights:
			# 	et2net['lt'][l][topt] += weight
			# for topl, weight in topl_weights:
			# 	et2net['tl'][t][topl] += weight
			words = [w for w in text if w in voca] # from text, only retain those words appearing in voca
			for w in words:
				nt2nodes['w'].add(w)
				et2net['tw'][t][w] += 1
				et2net['wt'][w][t] += 1
				et2net['wl'][w][l] += 1
				et2net['lw'][l][w] += 1
				# for topt, weight in topt_weights:
				# 	et2net['wt'][w][topt] += weight
				# for topl, weight in topl_weights:
				# 	et2net['wl'][w][topl] += weight
			for w1, w2 in itertools.combinations(words, r=2):
				et2net['ww'][w1][w2] += 1
				et2net['ww'][w2][w1] += 1

		# encode_continuous_proximity
		print "encoding_continuous_proximity"
		self.encode_continuous_proximity("ll", self.lClus, et2net, nt2nodes)
		self.encode_continuous_proximity("tt", self.tClus, et2net, nt2nodes)
		print "encoded_continuous_proximity"

		return nt2nodes, {et:et2net[et] for et in self.pd["etList"]}

	def encode_continuous_proximity(self, et, clus, et2net, nt2nodes):
		centers = clus.get_centers()
		nt = et[0]
		nodes = nt2nodes[nt]
		edges = []
		i = 0
		for n1, n2 in itertools.combinations(nodes, r=2):
			dist = np.linalg.norm(np.array(centers[int(n1)])-np.array(centers[int(n2)]))
			edges.append((n1, n2, dist))
		edges.sort(key=lambda tup:tup[2])
		for n1, n2, dist in edges[:len(nodes)*self.pd["nb_num"]]:
			if nt=='l':
				proximity = self.pd["kernel"](dist, self.pd["bandwidth_l"])
			else:
				proximity = self.pd["kernel"](dist, self.pd["bandwidth_t"])
			et2net[et][n1][n2] = proximity
			et2net[et][n2][n1] = proximity

	def predict(self, time, lat, lng, words):
		nt2vecs = self.nt2vecs
		location = [lat, lng]
		time = [self.pd["convert_ts"](time)]

		if not self.pd["kernel_candidate_num"]:
			l = self.lClus.predict(location)
			t = self.tClus.predict(time)
			ls_vec = nt2vecs['l'][l] if l in nt2vecs['l'] else np.zeros(self.pd["dim"])
			ts_vec = nt2vecs['t'][t] if t in nt2vecs['t'] else np.zeros(self.pd["dim"])
		else:
			l_vecs = [nt2vecs['l'][l]*weight for l, weight in self.lClus.tops(location) if l in nt2vecs['l']]
			t_vecs = [nt2vecs['t'][t]*weight for t, weight in self.tClus.tops(time) if t in nt2vecs['t']]
			ls_vec = np.average(l_vecs, axis=0) if l_vecs else np.zeros(self.pd["dim"])
			ts_vec = np.average(t_vecs, axis=0) if t_vecs else np.zeros(self.pd["dim"])

		w_vecs = [nt2vecs['w'][w] for w in words if w in nt2vecs['w']]
		ws_vec = np.average(w_vecs, axis=0) if w_vecs else np.zeros(self.pd["dim"])
		score = listCosine(ls_vec, ts_vec)+listCosine(ts_vec, ws_vec)+listCosine(ws_vec, ls_vec) #1
		return round(score, 6)

	def get_vec(self, query):
		nt2vecs = self.gsm2vec.nt2vecs
		if isinstance(query, str):
			return nt2vecs['w'][query.lower()]
		elif isinstance(query, list):
			return nt2vecs['l'][self.lClus.predict(query)]
		else:
			return nt2vecs['t'][self.tClus.predict([query])]

	def get_nbs1(self, query, nb_nt, neighbor_num=20):
		vec_query = self.get_vec(query)
		candidates = [(nb, listCosine(vec_query, vec_nb)) for nb, vec_nb in self.gsm2vec.nt2vecs[nb_nt].items()]
		candidates.sort(key=lambda tup:tup[1], reverse=True)
		return candidates[:neighbor_num]

	def get_nbs2(self, query1, query2, func, nb_nt, neighbor_num=20):
		vec_query1 = self.get_vec(query1)
		vec_query2 = self.get_vec(query2)
		candidates = [(nb, func(listCosine(vec_query1, vec_nb), listCosine(vec_query2, vec_nb))) \
			for nb, vec_nb in self.gsm2vec.nt2vecs[nb_nt].items()]
		candidates.sort(key=lambda tup:tup[1], reverse=True)
		return candidates[:neighbor_num]

	def convert_freq_to_pmi(self):
		pass


class LMeanshiftClus(object):
	def __new__(cls, pd):
		return MeanshiftClus(pd, pd["bandwidth_l"])

class TMeanshiftClus(object):
	def __new__(cls, pd):
		return MeanshiftClus(pd, pd["bandwidth_t"])

class MeanshiftClus:
	def __init__(self, pd, bandwidth):
		self.pd = pd
		self.bandwidth = bandwidth
		self.ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=5)

	def fit(self, X):
		X = np.array(X)
		self.ms.fit(X)
		return [str(label) for label in self.ms.labels_]

	def predict(self, x):
		return str(self.ms.predict([x])[0])

	def get_centers(self):
		return [list(center) for center in self.ms.cluster_centers_]

	def tops(self, x):
		candidates = []
		for clus, center in enumerate(self.ms.cluster_centers_):
			candidates.append( (str(clus), np.linalg.norm(np.array(x)-center)) )
		candidates.sort(key=lambda tup:tup[1])
		return [(candidate[0], self.pd["kernel"](candidate[1], self.bandwidth) ) for candidate in candidates[:self.pd["kernel_candidate_num"]]]

class LGridClus:
	def __init__(self, pd):
		self.grid_num = pd["grid_num"]

	def fit(self, locations):
		lats, lngs = zip(*locations)
		self.min_lat, self.min_lng = min(lats), min(lngs)
		self.max_lat, self.max_lng = max(lats), max(lngs)
		return [self.predict(location) for location in locations]

	def predict(self, location):
		lat, lng = location
		lat = min(max(self.min_lat, lat), self.max_lat)
		lng = min(max(self.min_lng, lng), self.max_lng)
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
	def __init__(self, pd):
		self.grid_num = pd["grid_num"]

	def fit(self, times):
		self.min, self.max = min([time[0] for time in times]), max([time[0] for time in times])
		return [self.predict(time) for time in times]

	def predict(self, time):
		time = time[0]
		time = min(max(self.min, time), self.max)
		grid_len = (self.max-self.min)/self.grid_num
		return str(int((time-self.min)/grid_len))

	def get_centers(self):
		grid_len = (self.max-self.min)/self.grid_num
		return [[self.min+i*grid_len] for i in range(self.grid_num)]

class TVoidClus:
	def __init__(self, pd):
		self.centers = []

	def fit(self, times):
		self.centers = times
		return [self.predict(time) for time in times]

	def predict(self, time):
		return str(time[0])

	def get_centers(self):
		return self.centers

class Gsm2vec_line:
	def __init__(self, pd):
		self.pd = pd
		self.nt2vecs = dict()
		self.io = IO()

	def fit(self, nt2nodes, et2net):
		self.write_line_input(nt2nodes, et2net)
		self.execute_line()
		self.read_line_output()
		return self.nt2vecs

	def write_line_input(self, nt2nodes, et2net):
		for nt, nodes in nt2nodes.items():
			print nt, len(nodes)
			node_file = open(self.io.line_dir+"node-"+nt+".txt", 'w')
			for node in nodes:
				node_file.write(node+"\n")
		all_et = [nt1+nt2 for nt1, nt2 in itertools.product(self.pd["ntList"], repeat=2)]
		for et in all_et:
			edge_file = open(self.io.line_dir+"edge-"+et+".txt", 'w')
			if et in et2net:
				for u, u_nb in et2net[et].items():
					for v, weight in u_nb.items():
						edge_file.write("\t".join([u, v, str(weight), "e"])+"\n")

	def execute_line(self):
		command = ["./hin2vec"]
		command += ["-size", str(self.pd["dim"])]
		command += ["-negative", str(self.pd["negative"])]
		command += ["-alpha", str(self.pd["alpha"])]
		command += ["-samples", str(self.pd["samples"])]
		command += ["-threads", str(self.pd["threads"])]
		command += ["-second_order", str(self.pd["second_order"])]
		call(command, cwd=self.io.line_dir)

	def read_line_output(self):
		for nt in self.pd["ntList"]:
			if nt!=self.pd["predict_type"] and self.pd["second_order"] and self.pd["use_context_vec"]:
				vecs_file = open(self.io.line_dir+"context-"+nt+".txt", 'r')
			else:
				vecs_file = open(self.io.line_dir+"output-"+nt+".txt", 'r')
			vecs = dict()
			for line in vecs_file:
				node, vec_str = line.strip().split("\t")
				vecs[node] = np.array([float(i) for i in vec_str.split(" ")])
			self.nt2vecs[nt] = vecs


class Gsm2vec:
	def __init__(self, pd):
		self.pd = pd
		self.nt2vecs = None
		self.nt2cvecs = None
		self.start_time = time.time()

	def fit(self, nt2nodes, et2net):
		pd = self.pd
		sample_size = int(pd["samples"]*1000000)
		# initialization not specified in the paper, got wrong at the beginning
		self.nt2vecs = {nt:{node:(np.random.rand(pd["dim"])-0.5)/pd["dim"] for node in nt2nodes[nt]} for nt in pd["ntList"]}
		self.nt2cvecs = {nt:{node:(np.random.rand(pd["dim"])-0.5)/pd["dim"] for node in nt2nodes[nt]} for nt in pd["ntList"]}
		et2optimizer = {et:self.Optimizer(et2net[et], pd, sample_size) for et in et2net}
		alpha = pd["alpha"]
		print 'sample', 'time', 'alpha', et2optimizer.keys()
		for i in range(sample_size):
			if i%1000==0 and pd["adaptive_lr"]:
				alpha = pd["alpha"] * (1 - float(i) / sample_size)
				if alpha < pd["alpha"]*0.0001:
					alpha = pd["alpha"]*0.0001
			if i%100000==0:
				print i, round(time.time()-self.start_time, 1), round(alpha, 5), [et2optimizer[et].get_objective(self.nt2vecs, et) for et in et2optimizer]
			for et in et2net:
				tu, tv = et[0], et[1]
				vecs_u, vecs_v = self.nt2vecs[tu], self.nt2vecs[tv]
				if pd["version"]==0 and pd["second_order"] or pd["version"]==1 and tv=='w' or pd["version"]==2 and tu=='w' and tv=='w':
					vecs_v = self.nt2cvecs[tv]
				et2optimizer[et].sgd_one_step(vecs_u, vecs_v, alpha)
		nt2vecs = dict()
		for nt in nt2nodes:
			if pd["version"]==0 and nt!=pd["predict_type"] and pd["second_order"] and pd["use_context_vec"]:
				nt2vecs[nt] = self.nt2cvecs[nt]
			else:
				nt2vecs[nt] = self.nt2vecs[nt]
		return nt2vecs
		

	class Optimizer:
		def __init__(self, net, pd, sample_size):
			self.pd = pd
			u2d = defaultdict(float)
			v2d = defaultdict(float)
			ranked_edges = []
			for u in net:
				for v in net[u]:
					u2d[u] += net[u][v]
					v2d[v] += net[u][v]
					ranked_edges.append(net[u][v])
			ranked_edges.sort(reverse=True)
			self.ns_thresh = ranked_edges[int((len(ranked_edges)-1)*self.pd["ns_refuse_percent"])]
			self.net = net
			self.u2samples = {u:iter(np.random.choice( net[u].keys(), 100, 
								p=self.normed(net[u].values()) )) for u in net}
			self.nega_samples = iter(np.random.choice( v2d.keys(), int(sample_size*1.4)*self.pd["negative"]*self.pd["ns_candidate_num"], 
								p=self.normed(np.power(v2d.values(), 0.75)) ))
			self.samples = iter(np.random.choice( u2d.keys(), sample_size, 
								p=self.normed(u2d.values()) ))
		
		def sgd_one_step(self, vecs_u, vecs_v, alpha):
			u = self.samples.next()
			try:
				v = self.u2samples[u].next()
			except StopIteration:
				self.u2samples[u] = iter(np.random.choice(self.net[u].keys(), 100, 
										p=self.normed(self.net[u].values()) ))
				v = self.u2samples[u].next()
			error_vec = np.zeros(self.pd["dim"])
			for j in range(self.pd["negative"]+1):
				if j==0:
					target = v
					label = 1
					f = np.dot(vecs_u[u], vecs_v[target])
				else:
					target, f = None, float('-inf')
					while not target:
						for i in range(self.pd["ns_candidate_num"]):
							target_candidate = self.nega_samples.next()
							if not (target_candidate in self.net[u] and self.net[u][target_candidate]>self.ns_thresh):
								f_candidate = np.dot(vecs_u[u], vecs_v[target_candidate])
								if f_candidate>f:
									target, f = target_candidate, f_candidate
					label = 0
				g = (label - expit(f)) * alpha
				error_vec += g*vecs_v[target]
				vecs_v[target] += g*vecs_u[u]
			vecs_u[u] += error_vec
	
		def get_objective(self, nt2vecs, et):
			objective = 0
			vecs_u, vecs_v = nt2vecs[et[0]], nt2vecs[et[1]]
			for u in self.net:
				for v in self.net[u]:
					try:
						f = np.dot(vecs_u[u], vecs_v[v])
						objective += self.net[u][v]*math.log(expit(f))
					except ValueError:
						print u,v,vecs_u[u],vecs_v[v],f,expit(f)
						exit(0)
			return -round(objective, 2)

		def normed(self, x):
			return x/np.linalg.norm(x, ord=1)


class Gsm2vec_relation:
	def __init__(self, pd):
		self.pd = pd
		self.nt2vecs = None
		self.nt2cvecs = None
		self.start_time = time.time()

	def fit(self, nt2nodes, relations):
		samples = int(self.pd["samples"]*1000000)
		self.nt2vecs = {nt:{node:(np.random.rand(self.pd["dim"])-0.5)/self.pd["dim"] for node in nt2nodes[nt]} for nt in self.pd["ntList"]}
		self.nt2cvecs = {nt:{node:(np.random.rand(self.pd["dim"])-0.5)/self.pd["dim"] for node in nt2nodes[nt]} for nt in self.pd["ntList"]}
		nt2vecs = self.nt2vecs
		nt2cvecs = self.nt2cvecs
		nt2nega_samples = {nt:iter(np.random.choice( nt2vecs[nt].keys(), samples*self.pd['negative'] )) for nt in nt2nodes}
		sample_cnt = 0
		while True:
			random.shuffle(relations)
			for relation in relations:
				sample_cnt += 1
				if sample_cnt%10000==0:
					print sample_cnt, time.time()-self.start_time
				if sample_cnt>samples:
					return
				alpha = self.pd["alpha"] * (1 - float(sample_cnt) / samples)
				if alpha < self.pd["alpha"]*0.0001:
					alpha = self.pd["alpha"]*0.0001
				sum_vec = np.zeros(self.pd["dim"])
				for nt in nt2nodes:
					for entity in relation[nt]:
						sum_vec += nt2cvecs[nt][entity]*relation[nt][entity]
				for nt in nt2nodes:
					for entity in relation[nt]:
						minus_entity_vec = sum_vec-nt2cvecs[nt][entity]
						for j in range(self.pd['negative']+1):
							if j==0:
								target = entity
								label = 1
							else:
								try:
									target = nt2nega_samples[nt].next()
								except StopIteration:
									nt2nega_samples[nt] = iter(np.random.choice( nt2vecs[nt].keys(), samples*self.pd['negative']))
									target = nt2nega_samples[nt].next()
								if target in relation[nt]: continue
								label = 0
							f = np.dot(nt2vecs[nt][target], minus_entity_vec)
							g = (label - expit(f)) * alpha
							nt2vecs[nt][target] += g*minus_entity_vec
							# for nt2 in nt2nodes:
							# 	for entity2 in relation[nt2]:
							# 		if not (nt==nt2 and entity==entity2):
							# 			nt2cvecs[nt2][entity2] += g*nt2vecs[nt][target]*relation[nt2][entity2]
		nt2vecs = dict()
		for nt in ntList:
			if nt!=self.pd["predict_type"] and self.pd["second_order"] and self.pd["use_context_vec"]:
				nt2vecs[nt] = self.nt2cvecs[nt]
			else:
				nt2vecs[nt] = self.nt2vecs[nt]
		return nt2vecs