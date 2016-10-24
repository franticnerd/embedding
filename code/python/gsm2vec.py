import numpy as np
import random
from sklearn.cluster import MeanShift
from collections import defaultdict
import time, datetime
from time import time as cur_time
from zutils.formula import listCosine
import itertools
import sys
from subprocess import call
from scipy.special import expit
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.neighbors import NearestNeighbors
sys.path.append('../')
from lgta.lgta import *
from mgtm.mgtm import *
sys.path.append('../scikit-tensor-master')
from sktensor import dtensor, cp_als

def convert_ts(ts):
	# return (ts/60)%(60*24*7)
	# return (ts/3600)%(24*7)
	# return (ts/3600)%24
	# return ts/3600
	# return (ts)%(3600*24)
	return (ts)%(3600*24*7)

class LgtaPredictor:
	def __init__(self, pd):
		self.lgta = LGTA("/shared/data/czhang82/embedding/"+pd['dataset']+"/lgta/")

	def fit(self, tweets, voca):
		pass

	def predict(self, time, lat, lng, words):
		return self.lgta.calc_probability(lat, lng, words)

	def gen_spatial_feature(self, lat, lng):
		return self.lgta.gen_spatial_feature(lat, lng)

	def gen_temporal_feature(self, time):
		return self.lgta.gen_temporal_feature(time)

	def gen_textual_feature(self, words):
		return self.lgta.gen_textual_feature(words)


class MgtmPredictor:
	def __init__(self, pd):
		self.mgtm = MGTM("/shared/data/czhang82/embedding/"+pd['dataset']+"/mgtm/")

	def fit(self, tweets, voca):
		pass

	def predict(self, time, lat, lng, words):
		return self.mgtm.calc_probability(lat, lng, words)

	def gen_spatial_feature(self, lat, lng):
		return self.mgtm.gen_spatial_feature(lat, lng)

	def gen_temporal_feature(self, time):
		return self.mgtm.gen_temporal_feature(time)

	def gen_textual_feature(self, words):
		return self.mgtm.gen_textual_feature(words)


class TfidfPredictor:
	def __init__(self, pd):
		self.pd = pd
		self.lClus = pd["lClus"](pd)
		self.tClus = pd["tClus"](pd)
		self.nt2nodes = None
		self.et2net = None

	def fit(self, tweets, voca):
		self.voca = voca
		self.nt2nodes, self.et2net = self.prepare_training_data(tweets, voca)

	def prepare_training_data(self, tweets, voca):
		nt2nodes = {nt:defaultdict(float) for nt in self.pd["ntList"]}
		et2net = {et:defaultdict(lambda : defaultdict(float)) for et in ['lt','lw','tw','tl','wl','wt']}
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times  = [[convert_ts(tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)

		for location, time, text, l, t in zip(locations, times, texts, ls, ts):
			nt2nodes['l'][l] += 1
			nt2nodes['t'][t] += 1
			et2net['lt'][l][t] += 1
			et2net['tl'][t][l] += 1
			words = [w for w in text if w in voca] # from text, only retain those words appearing in voca
			for w in words:
				nt2nodes['w'][w] += 1
				et2net['tw'][t][w] += 1
				et2net['wt'][w][t] += 1
				et2net['lw'][l][w] += 1
				et2net['wl'][w][l] += 1
		for et in et2net:
			net = et2net[et]
			rnet = et2net[et[::-1]]
			for u in net:
				for v in net[u]:
					# net[u][v] /= len(rnet[v])
					net[u][v] *= math.log( (len(net)+1)/len(rnet[v]) )
		return nt2nodes, et2net

	def predict(self, time, lat, lng, words):
		nt2nodes, et2net = self.nt2nodes, self.et2net
		location = [lat, lng]
		time = [convert_ts(time)]
		l = self.lClus.predict(location)
		t = self.tClus.predict(time)
		lw = [ et2net['lw'][l][w] for w in words ]
		tw = [ et2net['tw'][t][w] for w in words ]
		lw_score = sum(lw)/len(lw)
		tw_score = sum(tw)/len(tw)
		lt_score = et2net['lt'][l][t]
		score = lw_score+tw_score+lt_score
		return round(score, 6)

	def gen_spatial_feature(self, lat, lng):
		location = [lat, lng]
		l = self.lClus.predict(location)
		vec = [self.et2net['lw'][l][w] for w in self.voca]
		thresh = sorted(vec, reverse=True)[10]
		vec = [num if num>thresh else 0 for num in vec]
		return np.array(vec)

	def gen_temporal_feature(self, time):
		time = [convert_ts(time)]
		t = self.tClus.predict(time)
		vec = [self.et2net['lw'][t][w] for w in self.voca]
		thresh = sorted(vec, reverse=True)[10]
		vec = [num if num>thresh else 0 for num in vec]
		return np.array(vec)

	def gen_textual_feature(self, words):
		words = set(words)
		vec = [1 if w in words else 0 for w in self.voca]
		return np.array(vec)


class PmiPredictor:
	def __init__(self, pd):
		self.pd = pd
		self.lClus = pd["lClus"](pd)
		self.tClus = pd["tClus"](pd)
		self.nt2nodes = None
		self.et2net = None

	def fit(self, tweets, voca):
		self.voca = voca
		self.nt2nodes, self.et2net = self.prepare_training_data(tweets, voca)

	def prepare_training_data(self, tweets, voca):
		nt2nodes = {nt:defaultdict(float) for nt in self.pd["ntList"]}
		et2net = {et:defaultdict(lambda : defaultdict(float)) for et in ['lt','lw','tw']}
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times  = [[convert_ts(tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)

		for location, time, text, l, t in zip(locations, times, texts, ls, ts):
			nt2nodes['l'][l] += 1
			nt2nodes['t'][t] += 1
			et2net['lt'][l][t] += 1
			words = [w for w in text if w in voca] # from text, only retain those words appearing in voca
			for w in words:
				nt2nodes['w'][w] += 1
				et2net['tw'][t][w] += 1
				et2net['lw'][l][w] += 1
		for et in et2net:
			net = et2net[et]
			for u in net:
				for v in net[u]:
					# net[u][v] /= (nt2nodes[et[1]][v])
					# net[u][v] /= (nt2nodes[et[0]][u]*nt2nodes[et[1]][v])
					net[u][v] = math.log( (net[u][v]*len(tweets)) / (nt2nodes[et[0]][u]*nt2nodes[et[1]][v]) )
		return nt2nodes, et2net

	def predict(self, time, lat, lng, words):
		nt2nodes, et2net = self.nt2nodes, self.et2net
		location = [lat, lng]
		time = [convert_ts(time)]
		l = self.lClus.predict(location)
		t = self.tClus.predict(time)
		lw = [ et2net['lw'][l][w] for w in words ]
		tw = [ et2net['tw'][t][w] for w in words ]
		lw_score = sum(lw)/len(lw) if lw else 0
		tw_score = sum(tw)/len(tw) if tw else 0
		lt_score = et2net['lt'][l][t]
		score = lw_score+tw_score+lt_score
		return round(score, 6)

	def gen_spatial_feature(self, lat, lng):
		location = [lat, lng]
		l = self.lClus.predict(location)
		vec = [self.et2net['lw'][l][w] for w in self.voca]
		thresh = sorted(vec, reverse=True)[10]
		vec = [num if num>thresh else 0 for num in vec]
		return np.array(vec)

	def gen_temporal_feature(self, time):
		time = [convert_ts(time)]
		t = self.tClus.predict(time)
		vec = [self.et2net['lw'][t][w] for w in self.voca]
		thresh = sorted(vec, reverse=True)[10]
		vec = [num if num>thresh else 0 for num in vec]
		return np.array(vec)

	def gen_textual_feature(self, words):
		words = set(words)
		vec = [1 if w in words else 0 for w in self.voca]
		return np.array(vec)


class SvdPredictor:
	def __init__(self, pd):
		self.pd = pd
		self.lClus = pd["lClus"](pd)
		self.tClus = pd["tClus"](pd)
		self.nt2nodes = None
		self.et2net = None

	def fit(self, tweets, voca):
		self.voca = voca
		self.nt2nodes, self.et2net = self.prepare_training_data(tweets, voca)

	def prepare_training_data(self, tweets, voca):
		nt2nodes = {nt:defaultdict(float) for nt in self.pd["ntList"]}
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times  = [[convert_ts(tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)
		maxDim = max(len(self.lClus.get_centers()), len(self.tClus.get_centers()))
		et2net = {et:[defaultdict(float) for _ in range(maxDim)] for et in ['lt','lw','tw']}

		for location, time, text, l, t in zip(locations, times, texts, ls, ts):
			l, t = int(l), int(t)
			nt2nodes['l'][l] += 1
			nt2nodes['t'][t] += 1
			et2net['lt'][l][t] += 1
			words = [w for w in text if w in voca] # from text, only retain those words appearing in voca
			for w in words:
				nt2nodes['w'][w] += 1
				et2net['tw'][t][w] += 1
				et2net['lw'][l][w] += 1

		for et in ['lt','lw','tw']:
			vectorizer = DictVectorizer()
			X = vectorizer.fit_transform(et2net[et])
			U, Sigma, VT = randomized_svd(X, n_components=200,n_iter=5)
			X = np.dot(np.dot(U,np.diag(Sigma)),VT)
			net = vectorizer.inverse_transform(X)
			for u in range(len(net)):
				et2net[et][u] = defaultdict(float,net[u])

		return nt2nodes, et2net

	def predict(self, time, lat, lng, words):
		nt2nodes, et2net = self.nt2nodes, self.et2net
		location = [lat, lng]
		time = [convert_ts(time)]
		l = self.lClus.predict(location)
		t = self.tClus.predict(time)
		l, t = int(l), int(t)
		lw = [ et2net['lw'][l][w] for w in words ]
		tw = [ et2net['tw'][t][w] for w in words ]
		lw_score = sum(lw)/len(lw)
		tw_score = sum(tw)/len(tw)
		lt_score = et2net['lt'][l][t]
		score = lw_score+tw_score+lt_score
		return round(score, 6)

	def gen_spatial_feature(self, lat, lng):
		location = [lat, lng]
		l = int(self.lClus.predict(location))
		vec = [self.et2net['lw'][l][w] for w in self.voca]
		thresh = sorted(vec, reverse=True)[10]
		vec = [num if num>thresh else 0 for num in vec]
		return np.array(vec)

	def gen_temporal_feature(self, time):
		time = [convert_ts(time)]
		t = int(self.tClus.predict(time))
		vec = [self.et2net['lw'][t][w] for w in self.voca]
		thresh = sorted(vec, reverse=True)[10]
		vec = [num if num>thresh else 0 for num in vec]
		return np.array(vec)

	def gen_textual_feature(self, words):
		words = set(words)
		vec = [1 if w in words else 0 for w in self.voca]
		return np.array(vec)


class TensorPredictor:
	def __init__(self, pd):
		self.pd = pd
		self.lClus = pd["lClus"](pd)
		self.tClus = pd["tClus"](pd)
		self.nt2vecs = dict()
		self.lmbda = []

	def fit(self, tweets, voca):
		self.w2i= dict()
		for i, w in enumerate(voca):
			self.w2i[w] = i
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times  = [[convert_ts(tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)
		num_l, num_t, num_w = len(self.lClus.get_centers()), len(self.tClus.get_centers()), len(self.w2i)
		print 'nums', num_l, num_t, num_w
		T = np.zeros(shape=(num_l, num_t, num_w))
		for text, l, t in zip(texts, ls, ts):
			l, t = int(l), int(t)
			words = [self.w2i[w] for w in text if w in self.w2i] # from text, only retain those words appearing in voca
			for w in words:
				T[l, t, w] += 1.0
		P, fit, itr, exectimes = cp_als(dtensor(T), self.pd['tensor_rank'], init='random')
		self.nt2vecs['l'] = P.U[0]
		self.nt2vecs['t'] = P.U[1]
		self.nt2vecs['w'] = P.U[2]
		self.lmbda = P.lmbda

	def predict(self, time, lat, lng, words):
		ls_vec = self.gen_spatial_feature(lat, lng)
		ts_vec = self.gen_temporal_feature(time)
		ws_vec = self.gen_textual_feature(words)
		score = sum([ls_vec[i]*ts_vec[i]*ws_vec[i]*self.lmbda[i] for i in range(self.pd['tensor_rank'])])
		# score = listCosine(ls_vec, ts_vec)+listCosine(ts_vec, ws_vec)+listCosine(ws_vec, ls_vec) #1
		return round(score, 6)

	def gen_spatial_feature(self, lat, lng):
		location = [lat, lng]
		l = int(self.lClus.predict(location))
		return self.nt2vecs['l'][l]

	def gen_temporal_feature(self, time):
		time = [convert_ts(time)]
		t = int(self.tClus.predict(time))
		return self.nt2vecs['t'][t]

	def gen_textual_feature(self, words):
		w_vecs = [self.nt2vecs['w'][self.w2i[w]] for w in words if w in self.w2i]
		return np.average(w_vecs, axis=0) if w_vecs else np.zeros(self.pd["tensor_rank"])


class Gsm2vecPredictor:
	def __init__(self, pd):
		self.pd = pd
		self.lClus = pd["lClus"](pd)
		self.tClus = pd["tClus"](pd)
		self.nt2vecs = None
		self.start_time = cur_time()

	def fit(self, tweets, voca):
		gsm2vec = self.pd["gsm2vec"](self.pd)
		if isinstance(gsm2vec, Gsm2vec_relation):
			nt2nodes, relations = self.prepare_training_data_for_Gsm2vec_relation(tweets, voca)
			self.nt2vecs = gsm2vec.fit(nt2nodes, relations)
		else:
			nt2nodes, et2net = self.prepare_training_data(tweets, voca)
			self.nt2vecs = gsm2vec.fit(nt2nodes, et2net)
		self.nt2nodes = nt2nodes

	def prepare_training_data_for_Gsm2vec_relation(self, tweets, voca):
		nt2nodes = {nt:set() for nt in self.pd["ntList"]}
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times  = [[convert_ts(tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)

		relations = []
		for location, time, text, l, t in zip(locations, times, texts, ls, ts):
			relation = defaultdict(lambda : defaultdict(float))
			for l, weight in self.lClus.tops(location):
				nt2nodes['l'].add(l)
				relation['l'][l] = weight
			for t, weight in self.tClus.tops(time):
				nt2nodes['t'].add(t)
				relation['t'][t] = weight
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

	def prepare_training_data(self, tweets, voca):
		nt2nodes = {nt:set() for nt in self.pd["ntList"]}
		nt2node2degree = {nt:defaultdict(float) for nt in self.pd["ntList"]}
		all_et = [nt1+nt2 for nt1, nt2 in itertools.product(self.pd["ntList"], repeat=2)]
		et2net = {et:defaultdict(lambda : defaultdict(float)) for et in all_et}
		texts = [tweet.words for tweet in tweets]
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times  = [[convert_ts(tweet.ts)] for tweet in tweets]
		ls = self.lClus.fit(locations)
		ts = self.tClus.fit(times)
		print 'clustering_done:', cur_time()-self.start_time,

		for location, time, text, l, t in zip(locations, times, texts, ls, ts):
			nt2nodes['l'].add(l)
			nt2node2degree['l'][l] += 1
			nt2nodes['t'].add(t)
			nt2node2degree['t'][t] += 1
			et2net['lt'][l][t] += 1
			et2net['tl'][t][l] += 1
			words = [w for w in text if w in voca] # from text, only retain those words appearing in voca
			for w in words:
				nt2nodes['w'].add(w)
				nt2node2degree['w'][w] += 1
				et2net['tw'][t][w] += 1
				et2net['wt'][w][t] += 1
				et2net['wl'][w][l] += 1
				et2net['lw'][l][w] += 1
			for w1, w2 in itertools.combinations(words, r=2):
				if w1!=w2:
					et2net['ww'][w1][w2] += 1
					et2net['ww'][w2][w1] += 1
		print 'build_network_done:', cur_time()-self.start_time,

		# encode_continuous_proximity
		# print "encoding_continuous_proximity"
		self.encode_continuous_proximity("ll", self.lClus, et2net, nt2nodes)
		self.encode_continuous_proximity("tt", self.tClus, et2net, nt2nodes)
		print 'encode_continuous_proximity_done:', cur_time()-self.start_time,
		# print "encoded_continuous_proximity"

		# for et in et2net:
		# 	net = et2net[et]
		# 	rnet = et2net[et[::-1]]
		# 	for u in net:
		# 		for v in net[u]:
		# 			net[u][v] /= len(rnet[v])
		return nt2nodes, {et:et2net[et] for et in self.pd["etList"]}

	def encode_continuous_proximity(self, et, clus, et2net, nt2nodes):
		nt = et[0]
		if self.pd["kernel_nb_num_"+nt]>1:
			nodes = nt2nodes[nt]
			for n1 in nodes:
				center = clus.get_centers()[int(n1)]
				for n2, proximity in clus.tops(center):
					if n1!=n2:
						et2net[et][n1][n2] = proximity
						et2net[et][n2][n1] = proximity

	def gen_spatial_feature(self, lat, lng):
		nt2vecs = self.nt2vecs
		location = [lat, lng]
		if self.pd['version']==0 and self.pd["kernel_nb_num_l"]>10:
			l_vecs = [nt2vecs['l'][l]*weight for l, weight in self.lClus.tops(location) if l in nt2vecs['l']]
			ls_vec = np.average(l_vecs, axis=0) if l_vecs else np.zeros(self.pd["dim"])
		else:
			l = self.lClus.predict(location)
			ls_vec = nt2vecs['l'][l] if l in nt2vecs['l'] else np.zeros(self.pd["dim"])
		return ls_vec

	def gen_temporal_feature(self, time):
		nt2vecs = self.nt2vecs
		time = [convert_ts(time)]
		if self.pd['version']==0 and self.pd["kernel_nb_num_t"]>10:
			t_vecs = [nt2vecs['t'][t]*weight for t, weight in self.tClus.tops(time) if t in nt2vecs['t']]
			ts_vec = np.average(t_vecs, axis=0) if t_vecs else np.zeros(self.pd["dim"])
		else:
			t = self.tClus.predict(time)
			ts_vec = nt2vecs['t'][t] if t in nt2vecs['t'] else np.zeros(self.pd["dim"])
		return ts_vec

	def gen_textual_feature(self, words):
		nt2vecs = self.nt2vecs
		w_vecs = [nt2vecs['w'][w] for w in words if w in nt2vecs['w']]
		ws_vec = np.average(w_vecs, axis=0) if w_vecs else np.zeros(self.pd["dim"])
		return ws_vec

	def predict(self, time, lat, lng, words):
		ls_vec = self.gen_spatial_feature(lat, lng)
		ts_vec = self.gen_temporal_feature(time)
		ws_vec = self.gen_textual_feature(words)
		score = listCosine(ls_vec, ts_vec)+listCosine(ts_vec, ws_vec)+listCosine(ws_vec, ls_vec) #1
		return round(score, 6)

	def get_vec(self, query):
		nt2vecs = self.nt2vecs
		if isinstance(query, str):
			return nt2vecs['w'][query.lower()]
		elif isinstance(query, list):
			return nt2vecs['l'][self.lClus.predict(query)]
		else:
			return nt2vecs['t'][self.tClus.predict([query])]

	def get_nbs1(self, query, nb_nt, neighbor_num=10):
		vec_query = self.get_vec(query)
		candidates = [(nb, listCosine(vec_query, vec_nb)) for nb, vec_nb in self.nt2vecs[nb_nt].items()]
		candidates.sort(key=lambda tup:tup[1], reverse=True)
		return candidates[:neighbor_num]

	def get_nbs2(self, query1, query2, func, nb_nt, neighbor_num=10):
		vec_query1 = self.get_vec(query1)
		vec_query2 = self.get_vec(query2)
		candidates = [(nb, func(listCosine(vec_query1, vec_nb), listCosine(vec_query2, vec_nb))) \
			for nb, vec_nb in self.nt2vecs[nb_nt].items()]
		candidates.sort(key=lambda tup:tup[1], reverse=True)
		return candidates[:neighbor_num]


class LMeanshiftClus(object):
	def __new__(cls, pd):
		return MeanshiftClus(pd, pd["bandwidth_l"], pd["kernel_bandwidth_l"], pd["kernel_nb_num_l"])

class TMeanshiftClus(object):
	def __new__(cls, pd):
		return MeanshiftClus(pd, pd["bandwidth_t"], pd["kernel_bandwidth_t"], pd["kernel_nb_num_t"])

class MeanshiftClus:
	def __init__(self, pd, bandwidth, kernel_bandwidth, kernel_nb_num):
		self.pd = pd
		self.kernel_bandwidth = kernel_bandwidth
		self.ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=5)
		self.nbrs = NearestNeighbors(n_neighbors=kernel_nb_num, algorithm='ball_tree')

	def fit(self, X):
		X = np.array(X)
		self.ms.fit(X)
		self.nbrs.fit(self.ms.cluster_centers_)
		return [str(label) for label in self.ms.labels_]

	def predict(self, x):
		return str(self.ms.predict([x])[0])

	def get_centers(self):
		return [list(center) for center in self.ms.cluster_centers_]

	def tops(self, x):
		[distances], [indices] = self.nbrs.kneighbors([x])
		return [(str(index), self.kernel(distance,self.kernel_bandwidth)) for index, distance in zip(indices, distances)]

	def kernel(self, u, h=1.0):
		u /= h
		return 0 if u>1 else math.e**(-u*u/2)

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

	def fit(self, nt2nodes, et2net):
		self.write_line_input(nt2nodes, et2net)
		self.execute_line()
		self.read_line_output()
		return self.nt2vecs

	def write_line_input(self, nt2nodes, et2net):
		for nt, nodes in nt2nodes.items():
			print nt, len(nodes)
			node_file = open(self.pd["line_dir"]+"node-"+nt+"-"+self.pd["job_id"]+".txt", 'w')
			for node in nodes:
				node_file.write(node+"\n")
		all_et = [nt1+nt2 for nt1, nt2 in itertools.product(self.pd["ntList"], repeat=2)]
		for et in all_et:
			edge_file = open(self.pd["line_dir"]+"edge-"+et+"-"+self.pd["job_id"]+".txt", 'w')
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
		command += ["-job_id", str(self.pd["job_id"])]
		call(command, cwd=self.pd["line_dir"], stdout=open("stdout.txt","wb"))

	def read_line_output(self):
		for nt in self.pd["ntList"]:
			if nt!=self.pd["predict_type"] and self.pd["second_order"] and self.pd["use_context_vec"]:
				vecs_file = open(self.pd["line_dir"]+"context-"+nt+"-"+self.pd["job_id"]+".txt", 'r')
			else:
				vecs_file = open(self.pd["line_dir"]+"output-"+nt+"-"+self.pd["job_id"]+".txt", 'r')
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
		self.start_time = cur_time()

	def fit(self, nt2nodes, et2net):
		pd = self.pd
		sample_size = int(pd["samples"]*1000000)
		# initialization not specified in the paper, got wrong at the beginning
		self.nt2vecs = {nt:{node:(np.random.rand(pd["dim"])-0.5)/pd["dim"] for node in nt2nodes[nt]} for nt in nt2nodes}
		self.nt2cvecs = {nt:{node:(np.random.rand(pd["dim"])-0.5)/pd["dim"] for node in nt2nodes[nt]} for nt in nt2nodes}
		et2optimizer = {et:self.Optimizer(et2net[et], pd, sample_size) for et in et2net}
		alpha = pd["alpha"]
		print 'sample', 'time', 'alpha', et2optimizer.keys()
		for i in range(sample_size):
			if i%1000==0 and pd["adaptive_lr"]:
				alpha = pd["alpha"] * (1 - float(i) / sample_size)
				if alpha < pd["alpha"]*0.0001:
					alpha = pd["alpha"]*0.0001
			if i%100000==0:
				print i, round(cur_time()-self.start_time, 1), round(alpha, 5), [et2optimizer[et].get_objective(self.nt2vecs, et) for et in et2optimizer]
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
		self.start_time = cur_time()

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
					print sample_cnt, cur_time()-self.start_time
				if sample_cnt>samples:
					nt2vecs = dict()
					for nt in self.pd["ntList"]:
						if nt!=self.pd["predict_type"] and self.pd["second_order"] and self.pd["use_context_vec"]:
							nt2vecs[nt] = self.nt2cvecs[nt]
						else:
							nt2vecs[nt] = self.nt2vecs[nt]
					return nt2vecs
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
							for nt2 in nt2nodes:
								for entity2 in relation[nt2]:
									if not (nt==nt2 and entity==entity2):
										nt2cvecs[nt2][entity2] += g*nt2vecs[nt][target]*relation[nt2][entity2]