from zcode.core.datasets.twitter.tweet_database import TweetDatabase
from zcode.core.datasets.twitter.filters import TweetIdContainFilter
from zcode.core.datasets.foursquare.checkin_database import CheckinDatabase
from zcode.core.datasets.foursquare.venue_database import VenueDatabase
from zcode.core.datasets.twitter.mongo_tweet import TweetMongo
from zcode.projects.activity_filtering.twitter.word_distribution import WordEntropyProcessor
from collections import Counter
import sys




class MGTM_Preprocess:

    def extract_from_mongo(self, dns, port, db, col, data_dir, tweet_id_file=None):
        td = self.get_tweet_database(dns, port, db, col)
        self.filter_tweets_by_ids(td, tweet_id_file)
        td.trim_words_by_frequency(word_dict_file=None, freq_threshold=100000, infreq_threshold=50)
        tweets = td.tweets

        output_file = data_dir + 'tweets.txt'
        with open(output_file, 'w') as fout:
            fout.write(str(len(tweets)) + '\n')
            for t in tweets:
                s = str(t.location.lat) + ' ' + str(t.location.lng)
                for word in t.message.words:
                    s += ' ' + word
                fout.write(s + '\n')

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




if __name__ == '__main__':
    if len(sys.argv) < 2:
        dns = 'dmserv4.cs.illinois.edu'
        port = 11111
        db = 'foursquare'
        col = 'train'
        data_dir = '/Users/chao/Dropbox/Research/embedding/data/4sq/mgtm/'
        tweet_id_file = None
    else:
        dns = sys.argv[1]
        port = int(sys.argv[2])
        db = sys.argv[3]
        col = sys.argv[4]
        data_dir = sys.argv[5]
        tweet_id_file = None if len(sys.argv) == 6 else sys.argv[6]
    mp = MGTM_Preprocess()
    mp.extract_from_mongo(dns, port, db, col, data_dir, tweet_id_file)
