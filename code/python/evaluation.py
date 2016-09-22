import cPickle as pickle
from io_utils import IO
from baseline import *
from gsm2vec import *
import time
import paras
import bisect
import gsm2vec
import os
import folium
from collections import defaultdict
from zutils.twitter.tweet_database import TweetDatabase as DB

io = IO('../run/'+paras.pd['dataset']+'.yaml')

class QuantitativeEval:
	def __init__(self,predictor):
		self.predictor = predictor

	def computeMRR(self,tweets,pd):
		start_time = time.time()
		predictType = pd["predict_type"]
		fake_num=pd["fake_num"]
		ranks,rranks = [],[]
		noiseList = np.random.choice(tweets,fake_num*len(tweets)).tolist()
		for tweet in tweets:
			scored_tweets = []
			score = self.predictor.predict(tweet.ts, tweet.lat, tweet.lng, tweet.words)
			scored_tweets.append((score,tweet))
			for i in range(fake_num):
				noise = noiseList.pop()
				if predictType=='l':
					noise_score = self.predictor.predict(tweet.ts, noise.lat, noise.lng, tweet.words)
				elif predictType=='t':
					noise_score = self.predictor.predict(noise.ts, tweet.lat, tweet.lng, tweet.words)
				else:
					noise_score = self.predictor.predict(tweet.ts, tweet.lat, tweet.lng, noise.words)
				scored_tweets.append((noise_score,noise))
			scored_tweets.sort()
			scores = zip(*scored_tweets)[0]
			# handle ties
			rank = len(scores)+1-(bisect.bisect_left(scores,score)+bisect.bisect_right(scores,score)+1)/2.0
			ranks.append(rank)
			rranks.append(1.0/rank)

			# if rank==11:
			# 	directory = io.output_dir+"bad_cases/"+str(len(ranks))+"/"
			# 	if not os.path.isdir(directory):
			# 		os.mkdir(directory)
			# 	map_osm = folium.Map(location=[tweet.lat, tweet.lng])
			# 	folium.Marker(location=[tweet.lat, tweet.lng], popup=str(rank)).add_to(map_osm)
			# 	folium.LatLngPopup().add_to(map_osm)
			# 	map_osm.save(directory+'location.html')
			# 	tweets_file = open(directory+"tweets.txt",'w')
			# 	tweets_file.write(str(tweet.ts/3600%24)+'\t'+tweet.text.encode('utf-8')+'\n')
			# 	for score, tweet in scored_tweets[::-1]:
			# 		tweets_file.write('\t'.join((str(score), tweet.text.encode('utf-8'), str(tweet.words)))+'\n')
				

		mrr,mr = sum(rranks)/len(rranks),sum(ranks)/len(ranks)
		evalFile = open(io.eval_file,'a')
		evalFile.write("time for testing: "+str(time.time()-start_time)+"\n")
		evalFile.write(str(datetime.datetime.now())+"\n")
		evalFile.write(paras.pd2string(pd))
		evalFile.write("node nums: "+str({nt:len(self.predictor.nt2nodes[nt]) for nt in pd["ntList"]})+"\n")
		evalFile.write("mrr,mr: "+str((mrr,mr))+"\n\n")
		return mrr, mr


class QualitativeEval:
	def __init__(self,predictor):
		self.predictor = predictor

	def printAndScribe(self,directory,ws,ts,ls):
		if not os.path.isdir(directory):
			os.mkdir(directory)
		print ws
		print ts
		open(directory+"words.txt",'w').write(str(ws)+"\n")
		open(directory+"times.txt",'w').write(str(ts)+"\n")
		map_osm = None
		centers = self.predictor.lClus.get_centers()
		for rank,l in enumerate(ls):
			center = centers[int(l)]
			if not map_osm:
				map_osm = folium.Map(location=center)
			folium.CircleMarker(location=center, radius=200).add_to(map_osm)
		folium.LatLngPopup().add_to(map_osm)
		map_osm.save(directory+'/locations.html')

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

def train(tweets,pd):
	start_time = time.time()
	predictor = pd["predictor"](pd)
	predictor.fit(tweets, get_voca(tweets)) 
	# pickle.dump(predictor,open(io.models_dir+'gsm2vecPredictor.model','w'))
	evalFile = open(io.eval_file,'a')
	evalFile.write("time for training: "+str(time.time()-start_time)+"\n")
	return predictor

def get_voca(tweets):
	if paras.pd['dataset']=='la':
		word_localness = pickle.load(open(IO().models_dir+'word_localness.model', 'r'))
		voca = set(zip(*word_localness)[0])
		voca.remove("")
	else:
		word2freq = defaultdict(int)
		for tweet in tweets:
			for word in tweet.words:
				word2freq[word] += 1
		word_and_freq = word2freq.items()
		word_and_freq.sort(reverse=True, key=lambda tup:tup[1])
		voca = set(zip(*word_and_freq[:10000])[0])
	return voca

def read_tweets():
	if paras.pd['dataset']=='la':
		tweets = pickle.load(open(io.models_dir+'act_tweets_'+'1000000'+'.model','r'))
		# tweets = pickle.load(open(io.models_dir+'act_tweets_'+str(paras.pd["data_size"])+'.model','r'))
		random.shuffle(tweets)
		trainSize = int(len(tweets)*paras.pd["train_ratio"])
		tweets_train, tweets_test = tweets[:trainSize], tweets[trainSize:]
	else:
		train_db = DB(io.clean_text_file, io.dns, io.port, io.db, 'train', io.index)
		tweets_train = [tweet for tweet in train_db.get_tweets_from_db()]
		test_db = DB(io.clean_text_file, io.dns, io.port, io.db, 'test', io.index)
		tweets_test = [tweet for tweet in test_db.get_tweets_from_db()]
	return tweets_train[:paras.pd["data_size"]], tweets_test[:10000]

def main(job_id, params):
	start_time = time.time()
	params = {k:v[0] for k,v in params.items()}
	print params

	pd = dict(paras.pd)
	for para in params:
		pd[para] = params[para]
	pd["job_id"] = str(job_id)

	rand_seed = pd["rand_seed"]
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	
	tweets_train, tweets_test = read_tweets()
	predictor = train(tweets_train, pd)
	mrr, mr = QuantitativeEval(predictor).computeMRR(tweets_test, pd)
	print "mr:", mr
	print "mrr:", mrr
	print "time:", time.time()-start_time
	return -mrr


if __name__ == '__main__':
	main(0, dict())

	# pd = dict(paras.pd)

	# rand_seed = pd["rand_seed"]
	# np.random.seed(rand_seed)
	# random.seed(rand_seed)
	# tweets = pickle.load(open(io.models_dir+'act_tweets_'+'100000'+'.model','r'))
	# # tweets = pickle.load(open(io.models_dir+'act_tweets_'+str(pd["data_size"])+'.model','r'))
	# random.shuffle(tweets)
	# trainSize = int(len(tweets)*pd["train_ratio"])
	# # tweets_train, tweets_test = tweets[:trainSize][:pd["data_size"]], tweets[trainSize:][:1000]
	# # predictor = train(tweets_train,pd)
	# # QuantitativeEval(predictor).computeMRR(tweets_test,pd)

	# predictor = train(tweets[:trainSize], pd)
	# QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)
	
	# predictor = pickle.load(open(io.models_dir+'gsm2vecPredictor.model','r'))
	# QualitativeEval(predictor).getNbs1('dodger')
	# QualitativeEval(predictor).getNbs2("beach", [34.008, -118.4961], lambda a,b:a-b)
	
	# for negative in [0,1,2,5,10,20]:
	# 	pd = dict(paras.pd)
	# 	pd['negative'] = negative
	# 	predictor = train(tweets[:trainSize], pd)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)
	# for alpha in [0.01, 0.025, 0.05, 0.1, 0.2]:
	# 	pd = dict(paras.pd)
	# 	pd['alpha'] = alpha
	# 	predictor = train(tweets[:trainSize], pd)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)
	# for samples in [1,5,10,25,50,100]:
	# 	pd = dict(paras.pd)
	# 	pd['samples'] = samples
	# 	predictor = train(tweets[:trainSize], pd)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)

	# for predict_type in ['w','l','t']:
		# for bandwidth_l in [0.03,0.01,0.005]:
			# pd = dict(paras.pd)
			# pd['bandwidth_l'] = bandwidth_l
			# pd["predict_type"] = predict_type
			# predictor = train(tweets[:trainSize], pd)
			# QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)
		# for bandwidth_t in [3000,1000,500]:
		# 	pd = dict(paras.pd)
		# 	pd['bandwidth_t'] = bandwidth_t
		# 	pd["predict_type"] = predict_type
		# 	predictor = train(tweets[:trainSize], pd)
		# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)

	# for predict_type in ['w','l','t']:
	# 	pd = dict(paras.pd)
	# 	pd["predict_type"] = predict_type
	# 	predictor = train(tweets[:trainSize], pd)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)

	# for grid_num in [50]:
	# 	pd = dict(paras.pd)
	# 	pd['grid_num'] = grid_num
	# 	predictor = train(tweets[:trainSize], pd)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)

	# for version in [0, 1, 2]:
	# 	pd = dict(paras.pd)
	# 	pd['version'] = version
	# 	predictor = train(tweets[:trainSize], pd)
	# 	QuantitativeEval(predictor).computeMRR(tweets[trainSize:], pd)

