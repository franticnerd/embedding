from zcode.core.datasets.twitter.tweet_database import TweetDatabase
from zcode.core.datasets.twitter.filters import TweetIdContainFilter
from zcode.core.datasets.foursquare.checkin_database import CheckinDatabase
from zcode.core.datasets.foursquare.venue_database import VenueDatabase
from zcode.core.datasets.twitter.mongo_tweet import TweetMongo
from zcode.projects.activity_filtering.twitter.word_distribution import WordEntropyProcessor
from collections import Counter
import sys

class LGTA_Preprocess:

    def extract_from_mongo(self, dns, port, db, col, data_dir, tweet_id_file=None):
        td = self.get_tweet_database(dns, port, db, col)
        self.filter_tweets_by_ids(td, tweet_id_file)
        td.trim_words_by_frequency(word_dict_file=None, freq_threshold=100000, infreq_threshold=50)
        tweets = td.tweets

        data_dir += 'input/'
        word_freq_file = data_dir + 'word_map.txt'
        word_dict = self.build_word_dict(tweets, word_freq_file)
        tweet_location_file = data_dir + 'tweet_locations.txt'
        self.write_tweet_locations(tweets, tweet_location_file)
        tweet_word_file = data_dir + 'tweet_words.txt'
        self.write_tweet_words(tweets, word_dict, tweet_word_file)


    def get_tweet_database(self, dns, port, db, col):
        tm = TweetMongo(dns, port, db, col)
        tweets = tm.get_all_tweets()
        td = TweetDatabase()
        td.set_tweets(tweets)
        print 'Finished loading # tweets', len(td.tweets)
        return td


    def filter_tweets_by_ids(self, td, tweet_id_file):
        if tweet_id_file is not None:
            preserve_tweet_ids = self.load_tweet_ids(tweet_id_file)
            tcf = TweetIdContainFilter(preserve_tweet_ids)
            td.apply_tweet_filter([tcf])
        print '# tweets after id-filtering:', len(td.tweets)


    def load_tweet_ids(self, tweet_id_file):
        preserve_tweet_ids = set()
        with open(tweet_id_file, 'r') as fin:
            for line in fin:
                preserve_tweet_ids.add(long(line.strip()))
        return preserve_tweet_ids



    def build_word_dict(self, tweets, word_freq_file):
        c = Counter()
        for t in tweets:
            c.update(t.message.words)
        word_freq = c.most_common()
        with open(word_freq_file, 'w') as fout:
            for w, c in word_freq:
                fout.write(w + ' ' + str(c) + '\n')
        ret = {}
        for i, e in enumerate(word_freq):
            word, freq = e[0], e[1]
            ret[word] = i + 1
            i += 1
        return ret


    def write_tweet_locations(self, tweets, tweet_location_file):
        with open(tweet_location_file, 'w') as fout:
            for t in tweets:
                s = str(t.location.lng) + ' ' + str(t.location.lat)
                fout.write(s + '\n')


    def write_tweet_words(self, tweets, word_dict, tweet_word_file):
        with open(tweet_word_file, 'w') as fout:
            tid = 1
            for t in tweets:
                for word in t.message.words:
                    word_id = word_dict[word]
                    fout.write(str(tid) + ' ' + str(word_id) + ' 1\n')
                tid += 1



if __name__ == '__main__':
    if len(sys.argv) < 2:
        dns = 'dmserv4.cs.illinois.edu'
        port = 11111
        db = 'foursquare'
        col = 'train'
        data_dir = '/Users/chao/Dropbox/Research/embedding/data/4sq/lgta/'
        tweet_id_file = None
    else:
        dns = sys.argv[1]
        port = int(sys.argv[2])
        db = sys.argv[3]
        col = sys.argv[4]
        data_dir = sys.argv[5]
        tweet_id_file = None if len(sys.argv) == 6 else sys.argv[6]
    lp = LGTA_Preprocess()
    lp.extract_from_mongo(dns, port, db, col, data_dir, tweet_id_file)

