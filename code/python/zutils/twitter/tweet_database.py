import sys
import pymongo as pm
from tweet_handler import Tweet
from space import GridSpace
from word_distribution import Distribution
from math import log
import operator
import codecs

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
    Read and clean tweets
    '''
    # get tweets from mongo db
    def get_tweets_from_db(self, query=None):
        cnt = 0
        for o in self.tweet_col.find(query):
            tweet = Tweet()
            tweet.load_from_mongo(o)
            cnt += 1
            if cnt % 10000 == 0:
                print 'Loaded %d tweets from mongodb.' % cnt
            yield tweet

    # read the raw tweet file line by line
    def get_tweets_from_file(self):
        with open(self.tweet_file, 'r') as fin:
            for line in fin:
                tweet = Tweet()
                try:
                    tweet.load(line)
                except:
                    continue
                yield tweet


    # write tweets from the tweet file into the mongo db.
    def dump_tweet_file_to_mongo(self, min_word_count=3):
        self.remove_tweets_from_db()  # remove all the tweets from db
        cnt, batch = 0, []
        # get the set of inserted tweet ids to avoid duplicates
        inserted_tweets = self.get_mongodb_tweet_ids()
        for tweet in self.get_tweets_from_file():
            if tweet.id in inserted_tweets or tweet.num_words() < min_word_count:
                continue
            batch.append(tweet.to_dict())
            inserted_tweets.add(tweet.id)
            cnt += 1
            if len(batch) == 10000:
                self.tweet_col.insert(batch)
                batch = []
                print 'Finished processing', cnt, 'tweets'
        self.tweet_col.insert(batch)
        print 'Done processing', cnt, 'tweets'
        # creat index
        self.tweet_col.ensure_index("timestamp")
        self.tweet_col.ensure_index("lat")
        self.tweet_col.ensure_index("lng")
        self.tweet_col.ensure_index("id")


    # read num tweets from the raw tweet file, and write to a json file
    def dump_tweets_file_to_json(self, json_file, num=sys.maxint):
        cnt = 0
        with open(json_file, 'w') as fout:
            for tweet in self.get_tweets_from_file():
                fout.write(tweet.to_json_string() + '\n')
                cnt += 1
                if cnt % 10000 == 0:
                    print 'Dumped json files for %d tweets.' % cnt
                if cnt == num:
                    break
        print 'Finished writing %d tweets to json.' % cnt

    # dump num clean tweets into the output file
    def dump_clean_tweets_to_file(self, output_file, num=sys.maxint, min_word_count=3):
        cnt = 0
        with open(output_file, 'w') as fout:
            for tweet in self.get_tweets_from_file():
                if tweet.num_words() < min_word_count:
                    continue
                fout.write(tweet.to_string() + '\n')
                cnt += 1
                if cnt % 10000 == 0:
                    print 'Dumped clean tweets for %d tweets.' % cnt
                if cnt == num:
                    break
        print 'Finished dumping %d clean tweets.' % cnt

    # read clean tweets
    def read_clean_tweets_from_file(self, clean_tweet_file):
        with open(clean_tweet_file, 'r') as fin:
            for line in fin:
                tweet = Tweet()
                tweet.parse_clean_tweet(line)
                yield tweet


    # dump clean text for num tweets into the output file
    # when training doc2vex, prefix needs to be used
    def dump_tweet_text(self, output_file, prefix=False, num=sys.maxint):
        cnt = 0
        with open(output_file, 'w') as fout:
            for tweet in self.get_tweets_from_file():
                # if there are too few words in a tweet, ignore it.
                if tweet.num_words() < 2:  continue
                text = tweet.get_clean_words_string() if prefix == False else \
                        '_*' + str(cnt) + ' ' + tweet.get_clean_words_string()
                fout.write(text + '\n')
                cnt += 1
                if cnt % 10000 == 0:
                    print 'Dumped text for %d tweets.' % cnt
                if cnt == num:
                    break
        print 'Finished dumping text for %d tweets.' % cnt

    # input: set of tweet ids, fetch tweets from db and write to file
    def write_tweets_to_file(self, tweet_ids, out_file):
        print 'Need to write %d tweets' %  len(tweet_ids)
        num_written = 0
        with codecs.open(out_file, 'w', 'utf-8') as fout:
            for tid in tweet_ids:
                tweet = self.get_one_tweet(tid)
                fout.write(','.join([str(tweet.lat), str(tweet.lng), tweet.created_at, tweet.text]) + '\n')
                num_written += 1
                if num_written % 1000 == 0:
                    print 'Finished writing %d / %d tweets' % (num_written, len(tweet_ids))
        print 'Finished writing %d tweets' % len(tweet_ids)

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
    # compute the spatiotemporal distributions for words
    def get_word_distributions(self, grid_bin_list):
        grid = self.__init_grid(grid_bin_list)
        vocab_vector = {}  # key: word, value: spatiotemporal localness
        for tweet in self.get_tweets_from_db():
            self.update_vocab_vector(vocab_vector, grid, tweet, grid_bin_list)
        return vocab_vector


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

    def get_word_localness(self, grid_bin_list, localness_file, compute=True, min_frequency=200, freq_thresh=40000):
        if compute:
            vocab_vector = self.get_word_distributions(grid_bin_list)
            vocab_localness = self.compute_word_localness(vocab_vector, min_frequency, freq_thresh)
            self.write_localness(vocab_localness, localness_file)
        else:
            return self.load_localness(localness_file)

    # compute the kl divergence
    def compute_word_localness(self, vocab_vector, min_frequency, freq_thresh):
        vocab_localness = []
        for word, vector in vocab_vector.items():
            # remove too infrequent words
            frequecy = vector.get_l1_norm()
            if  frequecy < min_frequency:
                continue
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


if __name__ == '__main__':
    p = TweetDatabase('tweets.txt', 'dmserv4.cs.illinois.edu', 11111, 'sample', 'raw')
    for t in p.get_tweets_from_db():
        print t




# # train w2v model
# def train(self):
#     model = gensim.models.Word2Vec(self.checkins, size=200, window=8, min_count = 2, workers=4)
#     model.save('model.txt')
#     print model.most_similar(positive=['coffee'])
#     print model.similarity('coffee', 'bar')
#     print model.similarity('coffee', 'starbucks')
