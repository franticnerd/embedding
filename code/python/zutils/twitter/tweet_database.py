import sys
import pymongo as pm
from tweet_handler import Tweet
from space import GridSpace
from word_distribution import Distribution
from math import log
import operator
import codecs
from bson.objectid import ObjectId
import folium
from zutils import text_processor, timestamp
import random
import cPickle as pickle
from scipy.stats import entropy

class TweetDatabase:

    '''
    Initialization
    '''
    # dns: string, port: int
    def __init__(self, tweet_file, dns, port, db_name, tweet_col_name, index_col_name=None):
        # tweet file
        self.tweet_file = tweet_file
        # init database
        try:
            conn = pm.MongoClient(dns, port)
            self.db = conn[db_name]
            self.tweet_col = self.db[tweet_col_name]
            if index_col_name is not None:
                self.index_col = self.db[index_col_name]
        except:
            print 'Unable to connect to mongoDB.'

    '''
    Database operations
    '''
    # get one tweet by id, only support db source
    def get_one_tweet(self, tweet_id=None):
        if tweet_id is None:
            o = self.tweet_col.find_one()
        else:
            o = self.tweet_col.find_one({'id':int(tweet_id)})
        tweet = Tweet()
        tweet.load_from_mongo(o)
        return tweet

    def num_tweets_in_mongo(self):
        return self.tweet_col.count()

    # find the tweets that have already been inserted.
    def get_mongodb_tweet_ids(self):
        tweet_ids = set()
        for tweet in self.tweet_col.find(projection = ['id']):
            tweet_ids.add(tweet['id'])
        return tweet_ids

    # drop all the tweets from the mongodb
    def remove_tweets_from_db(self):
        self.tweet_col.drop()

    # spatiotemporal query
    def get_attribute_max(self, attribute='timestamp'):
        return self.tweet_col.find().sort([(attribute, -1)]).limit(1)[0][attribute]

    def get_attribute_min(self, attribute='timestamp'):
        return self.tweet_col.find().sort([(attribute, 1)]).limit(1)[0][attribute]


    '''
    Inverted Index
    '''
    # write inverted index to db. Key: word, value: list of tweets
    def build_inverted_index(self):
        vocab_to_list = self.__get_vocab_to_tweetids()
        data = []
        for word, l in vocab_to_list.items():
            data.append({'word': word, 'list': l})
        # write the dict into mongo db
        self.index_col.drop()
        self.index_col.insert(data)
        self.index_col.ensure_index('word')

    # construct inverted index for keywords
    def __get_vocab_to_tweetids(self):
        vocab_to_list = {}  # key: word, value: list of tweet ids
        for tweet in self.get_tweets_from_db():
            for word in set(tweet.words):
                l = vocab_to_list.get(word, [])
                l.append(tweet.id)
                vocab_to_list[word] = l
        return vocab_to_list

    def get_tweet_ids_containing_word(self, word):
        tweet_ids = self.index_col.find_one({'word': word})['list']
        return set(tweet_ids)

    def num_tweets_containing_word(self, word):
        tweet_ids = self.index_col.find_one({'word': word})['list']
        return len(tweet_ids)

    '''
    Word spatiotemporal Distribution and localness
    '''

    def __init_grid(self, grid_bin_list):
        min_lat, max_lat = self.get_attribute_min('lat'), self.get_attribute_max('lat')
        min_lng, max_lng = self.get_attribute_min('lng'), self.get_attribute_max('lng')
        min_ts, max_ts = self.get_attribute_min('timestamp'), self.get_attribute_max('timestamp')
        if len(grid_bin_list) == 2:
            grid_range_list = [(min_lat, max_lat), (min_lng, max_lng)]
        else:
            grid_range_list = [(min_lat, max_lat), (min_lng, max_lng), (min_ts, max_ts)]
        return GridSpace(grid_range_list, grid_bin_list)

    # update the vector using one tweet
    def update_vocab_vector(self, vocab_vector, grid, tweet, grid_bin_list):
        # get grid id for the triplet for the tweet
        if len(grid_bin_list) == 3:
            dim = grid.get_grid_id([tweet.lat, tweet.lng, tweet.ts])
        else:
            dim = grid.get_grid_id([tweet.lat, tweet.lng])
        # dim = grid.get_grid_id([tweet.lat, tweet.lng])
        for word in set(tweet.words):
            # update word spatiotemporal vector
            # vector = vocab_vector.get(word, SparseVector())
            vector = vocab_vector.get(word, Distribution())
            # vector.add_value(dim, 1)
            vector.add_value(dim, tweet.uid)
            vocab_vector[word] = vector

    # compute the kl divergence
    def compute_word_localness(self, vocab_vector, min_frequency, freq_thresh):
        vocab_localness = []
        for word, vector in vocab_vector.items():
            frequecy = vector.get_l1_norm()

            # remove too infrequent words
            # if  frequecy < min_frequency:
            #     continue

            # KL divergence
            if frequecy < freq_thresh:
                localness = log(frequecy) - vector.get_entropy()
            else:
                localness = log(freq_thresh) - vector.get_entropy()
            vocab_localness.append((word, localness, frequecy))
        vocab_localness.sort(key = operator.itemgetter(1), reverse=True)
        return vocab_localness

    def write_localness(self, vocab_localness, out_file):
        with open(out_file, 'w') as fout:
            for word, localness, frequency in vocab_localness:
                fout.write(word + ',' + str(localness) + ',' + str(frequency) + '\n')

    def load_localness(self, localness_file):
        vocab_localness = []
        with open(localness_file, 'r') as fin:
            for line in fin:
                items = line.strip().split(',')
                word, localness, frequency = items[0], float(items[1]), int(items[2])
                vocab_localness.append((word, localness, frequency))
        return vocab_localness

    # get localized tweets
    def write_activity_tweets(self, word_localness, fraction, out_file):
        tweet_ids = self.find_activity_tweet_ids(word_localness, fraction)
        self.write_tweets_to_file(tweet_ids, out_file)

    # a tweet is an activity tweet iff it has at least one high localness word
    def find_activity_tweet_ids(self, word_localness, fraction):
        tweet_ids = set()
        n_words = int(len(word_localness) * fraction)
        for i in xrange(n_words):
            word = word_localness[i][0]
            containing_tweet_ids = self.get_tweet_ids_containing_word(word)
            tweet_ids |= containing_tweet_ids
        return tweet_ids

    def write_nonactivity_tweets(self, word_localness, fraction, out_file):
        tweet_ids = self.find_nonactivity_tweet_ids(word_localness, fraction)
        self.write_tweets_to_file(tweet_ids, out_file)

    def find_nonactivity_tweet_ids(self, word_localness, fraction):
        tweet_ids = self.get_mongodb_tweet_ids()  # all the tweets in the mongo db
        n_words = int(len(word_localness) * (1 - fraction))  # potential activity tweets 
        for i in xrange(n_words):
            word = word_localness[i][0]
            containing_tweet_ids = self.get_tweet_ids_containing_word(word)
            tweet_ids -= containing_tweet_ids
            print len(tweet_ids)
        return tweet_ids


####################################################################################

    # get tweets from mongo db
    def get_tweets_from_db(self, query=None):
        cnt = 0
        for o in self.tweet_col.find(query):
            tweet = Tweet()
            tweet.load_from_mongo(o)
            cnt += 1
            if cnt % 100000 == 0:
                print 'Loaded %d tweets from mongodb.' % cnt
            yield tweet

    def get_tweets_phrases_from_db(self, query=None):
        cnt = 0
        phrases = []
        for o in self.tweet_col.find(query):
            cnt += 1
            if cnt % 100000 == 0:
                print 'Loaded %d tweets from mongodb.' % cnt
            phrases.append(o['phrases'])
        return phrases

    def add_phrases_field_to_db(self, input_file):
        with open(input_file, 'r') as fin:
            i = 0
            for line in fin:
                line = line.strip()
                fields = line.split("\t")
                _id,text = fields
                _id = ObjectId(_id)
                if self.tweet_col.count({"_id":_id})!=1:
                    print _id,line
                    break
                phrases = []
                phrase = ""
                openBracket = False
                for c in text:
                    if openBracket:
                        if c=="]":
                            openBracket = False
                            phrases.append(phrase)
                            phrase = ""
                        else:
                            phrase += c
                    else:
                        if c=="[":
                            openBracket = True
                        elif c==" ":
                            phrases.append(phrase)
                            phrase = ""
                        else:
                            phrase += c
                phrases.append(phrase)
                self.tweet_col.update_one({"_id":_id}, {'$set':{"phrases":phrases}})
                if i % 100000 == 0:
                    print 'Updated %d tweets from mongodb.' % i
                i += 1

    # compute the spatiotemporal distributions for words
    def get_word_distributions(self, grid_bin_list, min_frequency=100):
        grid = self.__init_grid(grid_bin_list)
        vocab_vector = {}  # key: word, value: spatiotemporal localness
        for tweet in self.get_tweets_from_db():
            self.update_vocab_vector(vocab_vector, grid, tweet, grid_bin_list)
        return {word:vector for word, vector in vocab_vector.items() if vector.get_l1_norm()>=min_frequency}

    def get_word_localness(self, grid_bin_list, localness_file, compute=True, min_frequency=100, freq_thresh=40000):
        if compute:
            vocab_vector = self.get_word_distributions(grid_bin_list)
            vocab_localness = self.compute_word_localness(vocab_vector, min_frequency, freq_thresh)
            self.write_localness(vocab_localness, localness_file)
            return vocab_localness,vocab_vector
        else:
            return self.load_localness(localness_file)

    def serialize_tweets_from_db(self):
        with codecs.open(self.tweet_file, 'w') as fout:
            for tweet in self.db.get_tweets_from_db():
                pickle.dump(tweet,fout)

    def deserialize_tweets_from_file(self):
        # with codecs.open("/Users/keyangzhang/Documents/UIUC/Research/Embedding/embedding/data/la/input/message.txt", 'r') as fin:
        with codecs.open(self.tweet_file, 'r') as fin:
            cnt = 0
            while True:
                # if cnt==10000:
                #     return
                cnt += 1
                try: 
                    tweet = pickle.load(fin)
                    yield tweet
                except:
                    return

    def count_tweets_containing_words(self,voca):
        count = 0
        for tweet in self.get_tweets_from_db():
            for word in tweet.words:
                if word in voca:
                    count += 1
                    break
        return count

    def get_tweets_containing_words(self,voca):
        tweets = []
        for tweet in self.get_tweets_from_db():
            for word in tweet.words:
                if word in voca:
                    tweets.append(tweet)
                    break
        return tweets

    def print_tweets_sample_containing_word(self,word):
        count = 0
        for tweet in self.get_tweets_from_db():
            if word in tweet.words:
                print tweet.text
                count += 1
            if count==50:
                return

####################################################################################

if __name__ == '__main__':
    from sklearn.feature_extraction import DictVectorizer
    from scipy.stats import entropy
    a = {1:0.5,2:0.5}
    b = {1:0.4,2:0.6}
    print entropy(a,b)
