from collections import defaultdict
from tweet_handler import Tweet, POI

# class IO:
#     def __init__(self):
#         self.root_dir = '/Users/chao/data/projects/embedding/tweets-dev/'
#         # if platform == 'darwin' else '/shared/data/kzhang53/embedding/'+paras.pd['dataset']+'/'
#         self.input_dir = self.root_dir+'input/'
#         self.output_dir = self.root_dir+'output/'
#         self.models_dir = self.root_dir+'models/'
#         self.tweet_file = self.input_dir+'tweets.txt'
#         self.poi_file = self.input_dir+'pois.txt'
#         self.case_study_dir = self.output_dir+'case_study/'


def read_tweets(tweet_file):
    tweets = []
    for line in open(tweet_file):
        tweet = Tweet()
        tweet.load_tweet(line.strip())
        tweets.append(tweet)
    return tweets


def get_voca(tweets, voca_min=0, voca_max=20000):
    word2freq = defaultdict(int)
    for tweet in tweets:
        for word in tweet.words:
            word2freq[word] += 1
    word_and_freq = word2freq.items()
    word_and_freq.sort(reverse=True, key=lambda tup:tup[1])
    # print 'get_voca', len(tweets), len(word2freq)
    voca = set(zip(*word_and_freq[voca_min:voca_max])[0])
    if '' in voca:
        voca.remove('')
    return voca


def read_pois(poi_file):
    pois = []
    for line in open(poi_file):
        fields = line.strip().split(',')
        if len(fields)<5:
            continue
        poi_id, lat, lng, cat, name = fields[0], float(fields[1]), float(fields[2]), fields[3], ','.join(fields[4:]).lower()
        pois.append(POI(poi_id, lat, lng, cat, name))
    return pois
