from zcode.core.datasets.foursquare.checkin_database import CheckinDatabase
from zcode.core.datasets.foursquare.venue_database import VenueDatabase
from zcode.core.datasets.twitter.filters import EmptyCategoryFilter
from zcode.core.datasets.twitter.mongo_tweet import TweetMongo
from zcode.projects.activity_filtering.twitter.word_distribution import WordEntropyProcessor


class MGTM_Preprocess:

    def extract_from_mongo(self, dns, port, db, col, data_dir):
        tm = TweetMongo(dns, port, db, col)
        tweets = tm.get_all_tweets()
        output_file = data_dir + 'tweets.txt'
        with open(output_file, 'w') as fout:
            fout.write(str(len(tweets)) + '\n')
            for t in tweets:
                s = str(t.location.lat) + ' ' + str(t.location.lng)
                for word in t.message.words:
                    s += ' ' + word
                fout.write(s + '\n')

if __name__ == '__main__':
    dns = 'dmserv4.cs.illinois.edu'
    port = 11111
    db = 'foursquare'
    col = 'train'
    data_dir = '/Users/chao/Dropbox/Research/embedding/data/4sq/mgtm/'
    mp = MGTM_Preprocess()
    mp.extract_from_mongo(dns, port, db, col, data_dir)

