import bisect
from embed import *

class QuantitativeEvaluator:
    def __init__(self, predict_type='w', fake_num=10):
        self.ranks = []
        self.predict_type = predict_type
        self.fake_num = fake_num
        if self.predict_type=='p':
            self.pois = io.read_pois()

    def get_ranks(self, tweets, predictor):
        noiseList = np.random.choice((self.pois if self.predict_type=='p' else tweets), self.fake_num*len(tweets)).tolist()
        for tweet in tweets:
            scores = []
            if self.predict_type=='p':
                score = predictor.predict(tweet.ts, tweet.poi_lat, tweet.poi_lng, tweet.words, tweet.category)
            else:
                score = predictor.predict(tweet.ts, tweet.lat, tweet.lng, tweet.words, tweet.category)
            scores.append(score)
            if self.predict_type=='c':
                for category in paras.pd['category_list']:
                    if category!=tweet.category:
                        noise_score = predictor.predict(tweet.ts, tweet.lat, tweet.lng, tweet.words, category)
                        scores.append(noise_score)
            else:
                for i in range(self.fake_num):
                    noise = noiseList.pop()
                    if self.predict_type in ['l','p']:
                        noise_score = predictor.predict(tweet.ts, noise.lat, noise.lng, tweet.words, tweet.category)
                    elif self.predict_type=='t':
                        noise_score = predictor.predict(noise.ts, tweet.lat, tweet.lng, tweet.words, tweet.category)
                    elif self.predict_type=='w':
                        noise_score = predictor.predict(tweet.ts, tweet.lat, tweet.lng, noise.words, tweet.category)
                    scores.append(noise_score)
            scores.sort()
            # handle ties
            rank = len(scores)+1-(bisect.bisect_left(scores,score)+bisect.bisect_right(scores,score)+1)/2.0
            self.ranks.append(rank)

    def compute_mrr(self):
        ranks = self.ranks
        rranks = [1.0/rank for rank in ranks]
        mrr,mr = sum(rranks)/len(rranks),sum(ranks)/len(ranks)
        return round(mrr,4), round(mr,4)
