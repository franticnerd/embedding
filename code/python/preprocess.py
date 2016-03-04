import sys
from zutils import parameter
from zutils.twitter.tweet_database import TweetDatabase as DB
from io_utils import IO
from gensim.models.doc2vec import Doc2Vec, LabeledSentence


class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            yield LabeledSentence(words=line.split(), tags=['SENT_%s' % uid])

class Preprocessor:

    def __init__(self, para_file):
        self.para = parameter.yaml_loader().load(para_file)
        self.io = IO(para_file)
        self.db = DB(self.io.raw_tweet_file, self.io.dns, self.io.port, self.io.db, self.io.tweet, self.io.index)

    def prepare_data(self):
        self.db.dump_tweet_text(self.io.clean_text_file, prefix=True)
        # self.db.dump_tweet_file_to_mongo()
        # self.db.build_inverted_index(self.io.index)
        # for t in self.db.get_tweets_from_db():
        #     print t.get_id(),
        # self.db.dump_tweets_file_to_json(self.io.clean_text_file)
        # print self.db.get_one_tweet()
        # self.db.build_inverted_index()


    def compute_word_entropy(self):
        vocab_entropy = self.db.get_word_localness(self.io.num_bins, self.io.entropy_file, compute=False)
        # self.db.write_activity_tweets(vocab_entropy, 0.0001, self.io.activity_file)
        self.db.write_nonactivity_tweets(vocab_entropy, 0.05, self.io.nonactivity_file)

    def embed_doc(self):
        documents = LabeledLineSentence(self.io.clean_text_file)
        model = Doc2Vec(documents, size=200, window=8, min_count=5, workers=4)
        model.save(self.io.doc2vec_file)
        print model.docvecs.most_similar('SENT_10')
        print model.docvecs.most_similar('SENT_100')
        print model.docvecs.most_similar('SENT_1000')
        print model.docvecs.most_similar('SENT_10000')
        print model.docvecs.most_similar('SENT_100000')

if __name__ == '__main__':
    p = Preprocessor(sys.argv[1])
    p.prepare_data()
    # p.compute_word_entropy()
    # p.embed_doc()
