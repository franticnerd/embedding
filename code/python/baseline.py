import time
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import hstack

class LRPredictor:
    def fit(self, tweets):
        X = [tweet.words for tweet in tweets]
        ST = [[tweet.ts, tweet.lat, tweet.lng] for tweet in tweets]
        # X = ['This is is first document.'.split(),'is is is first document.'.split()]
        # ST = [[1,2,3],[4,5,6]]
        X = CountVectorizer(preprocessor=(lambda x:x),tokenizer=(lambda x:x)).fit_transform(X)
        X = TfidfTransformer().fit_transform(X)
        X = hstack([X,ST])

    def predict(self,time,lat,lng,words):
    	return random.random()

class RandomPredictor:
    def fit(self, tweets):
    	pass

    def predict(self,time,lat,lng,words):
    	return random.random()


