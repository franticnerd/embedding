import numpy as np
from scipy.stats import multivariate_normal as nm
from math import exp

class MGTM:

    def __init__(self, data_dir):
        self.input_dir = data_dir
        self.load()

    def load(self):
        self.load_word_dict()
        self.load_regions()
        self.load_pwz()
        self.load_pzr()


    def load_word_dict(self):
        word_freq_file = self.input_dir + 'wordmap.txt'
        self.id_to_word = {}
        self.word_to_id = {}
        with open(word_freq_file, 'r') as fin:
            skip = fin.readline()
            for line in fin:
                items = line.strip().split()
                word, word_id = items[0], int(items[1])
                self.id_to_word[word_id] = word
                self.word_to_id[word] = word_id

    # priors of regions
    def load_regions(self):
        cluster_file = self.input_dir + 'cluster500.csv'
        clusters = np.loadtxt(open(cluster_file,'r'))
        tweet_file = self.input_dir + 'tweets.txt'
        locations = []
        with open(tweet_file, 'r') as fin:
            skip = fin.readline()
            for line in fin:
                items = line.strip().split(' ')
                lat, lng = float(items[0]), float(items[1])
                locations.append((lat, lng))
        self.location_clusters = zip(locations, clusters)


    def get_closest_region(self, lat, lng):
        ret = -1
        min_dist = float("inf")
        for loc, cluster_id in self.location_clusters:
            dist = (lat - loc[0])**2 + (lng - loc[1])**2
            if dist < min_dist:
                min_dist, ret = dist, cluster_id
        return ret


    def load_pwz(self):
        self.pwz = []
        pwz_file = self.input_dir + 'model-final.theta'
        with open(pwz_file, 'r') as fin:
            for line in fin:
                items = line.strip().split(' ')
                values = [float(i) for i in items]
                self.pwz.append(values)


    def load_pzr(self):
        self.pzr = []
        pzr_file = self.input_dir + 'model-final.pi_s'
        with open(pzr_file, 'r') as fin:
            for line in fin:
                items = line.strip().split(' ')
                values = [float(i) for i in items]
                self.pzr.append(values)

    def get_regional_topic_prob(self, region_id, words):
        topic_priors = self.pzr[region_id]
        n_topic = len(topic_priors)
        ret = 0
        for i in xrange(n_topic):
            ret += topic_priors[i] * self.get_topic_words_prob(i, words)
        return ret


    # def get_topic_words_prob(self, topic_id, words):
    #     ret = []
    #     word_dist = self.pwz[topic_id]
    #     for word in words:
    #         if word in self.word_to_id:
    #             word_id = self.word_to_id[word]
    #             ret.append(word_dist[word_id])
    #         else:
    #             ret.append(0)
    #     return np.mean(ret)


    def get_topic_words_prob(self, topic_id, words):
        ret = 1.0
        word_dist = self.pwz[topic_id]
        for word in words:
            word_id = self.word_to_id[word]
            ret *= word_dist[word_id]
        return ret ** (1.0 / len(words))

    def calc_probability(self, lat, lng, words):
        region_id = int(self.get_closest_region(lat, lng))
        return self.get_regional_topic_prob(region_id, words)


    def gen_spatial_feature(self, lat, lng):
        return [lat, lng]

    def gen_temporal_feature(self, ts):
        return [ts]

    def gen_textual_feature(self, words):
        ret = []
        n_topic = len(self.pzr[0])
        for topic_id in xrange(n_topic):
            ret.append(self.get_topic_words_prob(topic_id, words))
        return ret



if __name__ == '__main__':
    # mgtm = MGTM('/Users/chao/Dropbox/Research/embedding/data/4sq/mgtm/')
    # print mgtm.calc_probability(42.6413, -73.7781, ['jfk'])

    mgtm = MGTM('/Users/chao/Dropbox/Research/embedding/data/la/mgtm/')
    print mgtm.calc_probability(33.9416, -118.4085, ['studio'])
    print mgtm.gen_spatial_feature(33.9416, -118.4085)
    print mgtm.gen_temporal_feature(123.9)
    print mgtm.gen_textual_feature(['lax', 'jfk'])
