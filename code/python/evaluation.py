import cPickle as pickle
from io_utils import IO
from baseline import *
from gsm2vec import *
import time
import paras
import bisect
import gsm2vec
import os

pd = dict(paras.pd)

class QuantitativeEval:
	def __init__(self,predictor):
		self.predictor = predictor

	def computeMRR(self,tweets):
		start_time = time.time()
		predictType = pd["predict_type"]
		fake_num=pd["fake_num"]
		ranks,rranks = [],[]
		noiseList = np.random.choice(tweets,fake_num*len(tweets)).tolist()
		for tweet in tweets:
			scores = []
			score = self.predictor.predict(tweet.ts, tweet.lat, tweet.lng, tweet.words)
			scores.append(score)
			for i in range(fake_num):
				noise = noiseList.pop()
				if predictType=='l':
					noise_score = self.predictor.predict(tweet.ts, noise.lat, noise.lng, tweet.words)
				elif predictType=='t':
					noise_score = self.predictor.predict(noise.ts, tweet.lat, tweet.lng, tweet.words)
				else:
					noise_score = self.predictor.predict(tweet.ts, tweet.lat, tweet.lng, noise.words)
				scores.append(noise_score)
			scores.sort()
			# handle ties
			rank = len(scores)+1-(bisect.bisect_left(scores,score)+bisect.bisect_right(scores,score)+1)/2.0
			ranks.append(rank)
			rranks.append(1.0/rank)
		mrr,mr = sum(rranks)/len(rranks),sum(ranks)/len(ranks)
		print mrr,mr
		print "time for testing", time.time()-start_time
		evalFile = open(io.eval_file,'a')
		evalFile.write(str(datetime.datetime.now())+"\n")
		evalFile.write(paras.pd2string(pd))
		evalFile.write("time elapsed: "+str(time.time()-start_time)+"\n")
		evalFile.write("node nums: "+str({nt:len(self.predictor.gsm2vec.nt2vecs[nt]) for nt in pd["ntList"]})+"\n")
		evalFile.write("mrr,mr: "+str((mrr,mr))+"\n\n")


class QualitativeEval:
	def __init__(self,predictor):
		self.predictor = predictor

	def printAndScribe(self,directory,ws,ts,ls):
		import folium
		if not os.path.isdir(directory):
			os.mkdir(directory)
		print ws
		print ts
		open(directory+"paras.txt",'w').write(paras.pd2string(pd))
		open(directory+"words.txt",'w').write(str(ws)+"\n")
		open(directory+"times.txt",'w').write(str(ts)+"\n")
		map_osm = None
		centers = self.predictor.lClus.get_centers()
		for rank,l in enumerate(ls):
			center = centers[int(l)]
			if not map_osm:
				map_osm = folium.Map(location=center)
			map_osm.circle_marker(location=center, popup=str(rank), radius=200)
		map_osm.lat_lng_popover()
		map_osm.create_map(path=directory+'/locations.html')

	def getNbs1(self,query):
		directory = io.output_dir+"case_study/"+str(query)+"/"
		ws = zip(*self.predictor.get_nbs1(query,'w'))[0]
		ts = zip(*self.predictor.get_nbs1(query,'t',24))[0]
		ls = zip(*self.predictor.get_nbs1(query,'l',20))[0]
		self.printAndScribe(directory,ws,ts,ls)

	def getNbs2(self,query1,query2,func=lambda a,b:a+b):
		ws = zip(*self.predictor.get_nbs2(query1,query2,func,'w'))[0]
		ts = zip(*self.predictor.get_nbs2(query1,query2,func,'t',24))[0]
		ls = zip(*self.predictor.get_nbs2(query1,query2,func,'l',20))[0]
		directory = io.output_dir+"case_study/"+str(query1)+"-"+str(query2)+"/"
		self.printAndScribe(directory,ws,ts,ls)

def train(tweets):
	start_time = time.time()
	predictor = Gsm2vecPredictor(pd)
	predictor.fit(tweets[:trainSize])
	pickle.dump(predictor,open(io.models_dir+'gsm2vecPredictor.model','w'))
	print "time for training", time.time()-start_time
	return predictor

if __name__ == '__main__':
	io = IO()

	rand_seed = pd["rand_seed"]
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	tweets = pickle.load(open(io.models_dir+'act_tweets_'+str(pd["data_size"])+'.model','r'))
	trainSize = int(len(tweets)*pd["train_ratio"])

	# predictor = train(tweets)
	# QuantitativeEval(predictor).computeMRR(tweets[trainSize:])
	
	# predictor = pickle.load(open(io.models_dir+'gsm2vecPredictor.model','r'))
	# QualitativeEval(predictor).getNbs1('beach')
	# QualitativeEval(predictor).getNbs2("beach", [34.008, -118.4961], lambda a,b:a-b)
	
	# for negative in [0,1,2,5,10,20]:
	# 	pd = dict(paras.pd)
	# 	pd['negative'] = negative
	# 	predictor = train(tweets)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:])
	# for alpha in [0.01, 0.025, 0.05, 0.1, 0.2]:
	# 	pd = dict(paras.pd)
	# 	pd['alpha'] = alpha
	# 	predictor = train(tweets)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:])
	# for samples in [1,5,10,25,50,100]:
	# 	pd = dict(paras.pd)
	# 	pd['samples'] = samples
	# 	predictor = train(tweets)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:])

	# for predict_type in ['w','l','t']:
	# 	for bandwidth_l in [0.1,0.03,0.01,0.003]:
	# 		pd = dict(paras.pd)
	# 		pd['bandwidth_l'] = bandwidth_l
	# 		pd["predict_type"] = predict_type
	# 		predictor = train(tweets)
	# 		QuantitativeEval(predictor).computeMRR(tweets[trainSize:])
	# 	for bandwidth_t in [100,30,10,3]:
	# 		pd = dict(paras.pd)
	# 		pd['bandwidth_t'] = bandwidth_t
	# 		pd["predict_type"] = predict_type
	# 		predictor = train(tweets)
	# 		QuantitativeEval(predictor).computeMRR(tweets[trainSize:])

	for predict_type in ['w','l','t']:
		pd = dict(paras.pd)
		pd["predict_type"] = predict_type
		predictor = train(tweets)
		QuantitativeEval(predictor).computeMRR(tweets[trainSize:])

	# for nb_num in [3,10,30,100]:
	# 	pd = dict(paras.pd)
	# 	pd['nb_num'] = nb_num
	# 	predictor = train(tweets)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:])
	# for grid_num in [50]:
	# 	pd = dict(paras.pd)
	# 	pd['grid_num'] = grid_num
	# 	predictor = train(tweets)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:])

