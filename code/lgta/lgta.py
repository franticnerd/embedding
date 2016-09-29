import numpy as np
from scipy.stats import multivariate_normal as nm
from math import exp

class LGTA:

    def __init__(self, data_dir):
        self.input_dir = data_dir + 'input/'
        self.output_dir = data_dir + 'output/'
        self.load()

    def load(self):
        self.load_word_dict()
        self.load_priors()
        self.load_mu()
        self.load_sigma()
        self.load_pwz()
        self.load_pzr()


    def load_word_dict(self):
        word_freq_file = self.input_dir + 'word_map.txt'
        self.word_dict = {}
        self.id_to_word = {}
        with open(word_freq_file, 'r') as fin:
            word_id =  0
            for line in fin:
                items = line.strip().split()
                word = items[0]
                self.word_dict[word_id] = word
                self.id_to_word[word] = word_id
                word_id += 1

    # priors of regions
    def load_priors(self):
        prior_file = self.output_dir + 'priors.txt'
        self.priors = np.loadtxt(open(prior_file,'r'), delimiter=',')

    def load_pwz(self):
        pwz_file = self.output_dir + 'pwz.txt'
        self.pwz = np.loadtxt(open(pwz_file,'r'), delimiter=',')

    def load_pzr(self):
        pzr_file = self.output_dir + 'pzr.txt'
        self.pzr = np.loadtxt(open(pzr_file,'r'), delimiter=',')

    def load_mu(self):
        mu_file = self.output_dir + 'mu.txt'
        self.mu = np.loadtxt(open(mu_file,'r'), delimiter=',')

    def load_sigma(self):
        sigma_file = self.output_dir + 'sigma.txt'
        self.sigma = np.loadtxt(open(sigma_file,'r'), delimiter=',')

    def print_top_words(self):
        cnt = 0
        for a in self.pwz:
            print 'topic', cnt
            ind = a.argsort()[-10:][::-1]
            for i, f in zip(ind, a[ind]):
                print '\t', self.word_dict[i], f
            cnt += 1


    def get_gaussian_prob(self, mu, sigma, lat, lng):
        cov_matrix = [[sigma[0], sigma[1]], [sigma[2], sigma[3]]]
        return exp(nm.logpdf([lng, lat], mean = mu,  cov=cov_matrix))

    def get_regional_topic_prob(self, region_id, words):
        topic_priors = self.pzr[region_id]
        n_topic = len(topic_priors)
        ret = 0
        for i in xrange(n_topic):
            ret += topic_priors[i] * self.get_topic_words_prob(i, words)
        return ret

    def get_topic_words_prob(self, topic_id, words):
        ret = []
        word_dist = self.pwz[topic_id]
        for word in words:
            word_id = self.id_to_word[word]
            ret.append(word_dist[word_id])
        return max(ret)

    # def get_topic_words_prob(self, topic_id, words):
    #     ret = 1.0
    #     word_dist = self.pwz[topic_id]
    #     for word in words:
    #         word_id = self.id_to_word[word]
    #         ret *= word_dist[word_id]
    #     return ret ** (1.0 / len(words))

    def calc_probability(self, lat, lng, words):
        ret = 0
        n_region = self.pzr.shape[0]
        for i in xrange(n_region):
            ret += self.priors[i] \
                * self.get_gaussian_prob(self.mu[i], self.sigma[i], lat, lng) \
                * self.get_regional_topic_prob(i, words)
        return ret


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
    # for 4sq data set
    # lgta = LGTA('/Users/chao/Dropbox/Research/embedding/data/4sq/lgta/')
    # lgta.print_top_words()
    # print lgta.calc_probability(40.6413, -73.7781, ['airport'])
    # for la data set
    lgta = LGTA('/Users/chao/Dropbox/Research/embedding/data/la/lgta/')
    lgta.print_top_words()
    print lgta.calc_probability(33.9416, -118.4085, ['lax', 'universal'])
    print lgta.gen_spatial_feature(33.9416, -118.4085)
    print lgta.gen_temporal_feature(123.9)
    print lgta.gen_textual_feature(['lax', 'jfk'])

