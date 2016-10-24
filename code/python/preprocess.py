import sys
from zutils import parameter
from zutils import formula
from zutils.twitter.tweet_database import TweetDatabase as DB
from io_utils import IO
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import gensim
import time
import cPickle as pickle
import codecs
from sklearn import linear_model
import scipy
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import folium
from collections import Counter
import random

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            yield LabeledSentence(words=line.split(), tags=['SENT_%s' % uid])

class Preprocessor:

    def __init__(self, para_file):
        self.start_time = time.time()
        self.para = parameter.yaml_loader().load(para_file)
        self.io = IO(para_file)
        self.db = DB(self.io.clean_text_file, self.io.dns, self.io.port, self.io.db, self.io.tweet, self.io.index)

    def prepare_data(self):
        self.db.print_text_from_db(self.io.clean_text_file)
        # self.db.add_phrases_field_to_db(self.io.segmented_text_file)
        # self.db.dump_tweet_file_to_mongo()
        # self.db.build_inverted_index(self.io.index)
        # for t in self.db.get_tweets_from_db():
        #     print t.get_id(),
        # self.db.dump_tweets_file_to_json(self.io.clean_text_file)
        # print self.db.get_one_tweet()
        # self.db.build_inverted_index()

    def compute_word_entropy(self):
        vocab_entropy = self.db.get_word_localness(self.io.num_bins, self.io.entropy_file, compute=True)
        # self.db.write_activity_tweets(vocab_entropy, 0.0001, self.io.activity_file)
        # self.db.write_nonactivity_tweets(vocab_entropy, 0.05, self.io.nonactivity_file)

    def embed_doc(self):
        documents = LabeledLineSentence(self.io.clean_text_file)
        model = Doc2Vec(documents, size=200, window=8, min_count=5, workers=4)
        model.save(self.io.doc2vec_file)
        print model.docvecs.most_similar('SENT_10')
        print model.docvecs.most_similar('SENT_100')
        print model.docvecs.most_similar('SENT_1000')
        print model.docvecs.most_similar('SENT_10000')
        print model.docvecs.most_similar('SENT_100000')

    def test_word2vec(self):
        sentences = [['second', 'sentence'], ['first', 'sentence']]
        model = gensim.models.Word2Vec(sentences, min_count=1)
        print model.similarity('first', 'second')
        print model.similarity('second', 'sentence')
        print model.similarity('first', 'sentence')

    def bootstrap_nn(self,seedratio=0.005,stepratio=0.005,endratio=0.2):
        st_distr = pickle.load(open(self.io.models_dir+'st_distr.model','r'))
        word_localness = pickle.load(open(self.io.models_dir+'word_localness.model','r'))
        word2vec = pickle.load(open(self.io.models_dir+'word_vec.model','r'))
        v = zip(*word_localness)[0]
        print "loading done:", time.time()-self.start_time
        # intialize seeds
        seedsize = int(len(v)*seedratio)
        pos,unknown,neg = v[:seedsize],v[seedsize:-seedsize],v[-seedsize:]
        # bootstarp
        while len(pos)<len(v)*endratio:
            pos,unknown,neg = self.bootstrap_onestep_nn(stepratio,v,pos,unknown,neg,st_distr,word2vec,0)
            pos,unknown,neg = self.bootstrap_onestep_nn(stepratio,v,pos,unknown,neg,st_distr,word2vec,1)
            # pos,unknown,neg = self.bootstrap_onestep_nn(stepratio,v,pos,unknown,neg,st_distr,word2vec,0.5)
        pickle.dump(pos,open(self.io.models_dir+'act_words.model','w'))

    def bootstrap_onestep_nn(self,stepratio,v,pos,unknown,neg,st_distr,word2vec,st_weight):
        candidates = []
        for candidate in unknown:
            score = 0
            for word in pos:
                if st_weight:
                    score += st_weight*formula.dictCosine(st_distr[candidate].data,st_distr[word].data)
                if 1-st_weight:
                    score += (1-st_weight)*formula.listCosine(word2vec[candidate],word2vec[word])
            for word in neg:
                if st_weight:
                    score -= st_weight*formula.dictCosine(st_distr[candidate].data,st_distr[word].data)
                if 1-st_weight:
                    score -= (1-st_weight)*formula.listCosine(word2vec[candidate],word2vec[word])
            candidates.append((candidate,score))
            # print time.time()-self.start_time
        candidates.sort(key=lambda tup:tup[1], reverse=True)
        unknown = zip(*candidates)[0]
        stepsize = int(len(v)*stepratio)
        pos += unknown[:stepsize]
        neg += unknown[-stepsize:]
        unknown = unknown[stepsize:-stepsize]
        print st_weight, time.time()-self.start_time
        print "top:",candidates[:5]
        print "bottom:",candidates[-5:]
        return pos,unknown,neg

    def pick_act_tweets(self):
        act_words = pickle.load(open(self.io.models_dir+'act_words.model','r'))
        act_tweets = self.db.get_tweets_containing_words(set(act_words))
        pickle.dump(act_tweets,open(self.io.models_dir+'act_tweets.model','w'))

    def bootstrap_clf(self,seedratio=0.02,stepratio=0.002,endratio=0.05):
        word_localness = pickle.load(open(self.io.models_dir+'word_localness.model','r'))
        word2vec = pickle.load(open(self.io.models_dir+'word_vec.model','r'))
        v = zip(*word_localness)[0]
        # intialize seeds
        seedsize = int(len(v)*seedratio)
        pos,unknown,neg = v[:seedsize],v[seedsize:-seedsize],v[-seedsize:]
        # bootstarp
        while len(pos)<len(v)*endratio:
            pos,unknown,neg = self.bootstrap_onestep_clf(stepratio,v,pos,unknown,neg,word2vec)

        pickle.dump(pos,open(self.io.models_dir+'act_words.model','w'))
        print "classication done:", time.time()-self.start_time
        # print self.db.count_tweets_containing_words(set(pos))
        with open(self.io.output_dir+"act_words.txt", 'w') as fout:
            fout.write("seed:\n")
            for word in pos[:seedsize]:
                fout.write(word+"\n")
            fout.write("\nbootstarp:\n")
            for word in pos[seedsize:]:
                fout.write(word+"\n")

    def bootstrap_onestep_clf(self,stepratio,v,pos,unknown,neg,word2vec):
        X = [word2vec[word] for word in pos+neg]
        y = [1 for word in pos]+[0 for word in neg]
        clf = linear_model.LogisticRegression()
        clf.fit(X,y)
        candidates = []
        for word in unknown:
            score = clf.predict_proba([word2vec[word]])[0][1]
            candidates.append((word,score))
        # s = sorted(candidates,key=lambda tup:tup[1], reverse=True)
        # print scipy.stats.spearmanr(zip(*s)[0], zip(*candidates)[0])
        candidates.sort(key=lambda tup:tup[1], reverse=True)
        unknown = zip(*candidates)[0]
        stepsize = int(len(v)*stepratio)
        pos += unknown[:stepsize]
        neg += unknown[-stepsize:]
        unknown = unknown[stepsize:-stepsize]
        # print time.time()-self.start_time
        # print "top:",candidates[:5]
        # print "bottom:",candidates[-5:]
        return pos,unknown,neg

    def comp_distributions(self):
        word_localness,st_distr = self.db.get_word_localness(self.io.num_bins, self.io.entropy_file, compute=True)
        sentences = [tweet.words for tweet in self.db.get_tweets_from_db()]
        word2vec = gensim.models.Word2Vec(sentences, size=100, window=5, workers=4, min_count=10)

        word2vec.save(self.io.models_dir+'word2vec.model')
        pickle.dump(st_distr,open(self.io.models_dir+'st_distr.model','w'))
        pickle.dump(word_localness,open(self.io.models_dir+'word_localness.model','w'))

    def filter_word2vec(self):
        word_localness = pickle.load(open(self.io.models_dir+'word_localness.model','r'))
        word2vec = gensim.models.Word2Vec.load(self.io.models_dir+'word2vec.model')
        v = zip(*word_localness)[0]
        word_vec = dict()
        for word in v:
            word_vec[word] = word2vec[word]
        pickle.dump(word_vec,open(self.io.models_dir+'word_vec.model','w'))

    def sample_act_tweets(self,sample_size=100000):
        act_tweets = pickle.load(open(self.io.models_dir+'act_tweets.model','r'))
        random.seed(1)
        random.shuffle(act_tweets)
        pickle.dump(act_tweets[:sample_size],open(self.io.models_dir+'act_tweets_'+str(sample_size)+'.model','w'))

    def get_spatial_units(self):
        tweets = pickle.load(open(self.io.models_dir+'act_tweets.model','r'))
        print "loading done:", time.time()-self.start_time
        X = [[tweet.lat,tweet.lng] for tweet in tweets]
        X = np.array(X)
        # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100000)
        # print bandwidth, time.time()-self.start_time
        ms = MeanShift(bandwidth=0.005, bin_seeding=True, n_jobs=5)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        print len(cluster_centers), time.time()-self.start_time

        cluster2size = Counter(labels)
        map_osm = None
        for cluster,center in enumerate(cluster_centers):
            center = list(center)
            if not map_osm:
                map_osm = folium.Map(location=center)
            map_osm.circle_marker(location=center, popup=str(cluster2size[cluster]), radius=50)
        map_osm.lat_lng_popover()
        map_osm.create_map(path=self.io.output_dir+'cluster_centers.html')

    def test(self):
        tweets = pickle.load(open(self.io.models_dir+'act_tweets_100000.model','r'))
        print len(tweets)
        print time.time()-self.start_time

    def randomly_pick_tweets_as_training(self, num=950724):
        tweets = [tweet for tweet in self.db.get_tweets_from_db()]
        random.shuffle(tweets)
        pickle.dump(tweets[:num], open(self.io.models_dir+'random_training.model','w'))


if __name__ == '__main__':
    p = Preprocessor("../run/la.yaml")
    p.randomly_pick_tweets_as_training()
