from zcode.core.datasets.foursquare.checkin_database import CheckinDatabase
from zcode.core.datasets.foursquare.venue_database import VenueDatabase
from zcode.core.datasets.twitter.filters import EmptyCategoryFilter
from zcode.core.datasets.twitter.mongo_tweet import TweetMongo
from zcode.projects.activity_filtering.twitter.word_distribution import WordEntropyProcessor
from collections import Counter

class LGTA_Preprocess:

    def extract_from_mongo(self, dns, port, db, col, data_dir):
        tm = TweetMongo(dns, port, db, col)
        tweets = tm.get_all_tweets()
        data_dir += 'input/'

        word_freq_file = data_dir + 'word_map.txt'
        word_dict = self.build_word_dict(tweets, word_freq_file)

        tweet_location_file = data_dir + 'tweet_locations.txt'
        self.write_tweet_locations(tweets, tweet_location_file)

        tweet_word_file = data_dir + 'tweet_words.txt'
        self.write_tweet_words(tweets, word_dict, tweet_word_file)



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
    dns = 'dmserv4.cs.illinois.edu'
    port = 11111
    db = 'foursquare'
    col = 'train'
    data_dir = '/Users/chao/Dropbox/Research/embedding/data/4sq/lgta/'
    lp = LGTA_Preprocess()
    lp.extract_from_mongo(dns, port, db, col, data_dir)

