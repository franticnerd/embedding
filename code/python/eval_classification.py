from zutils.twitter.tweet_database import TweetDatabase as DB
from io_utils import IO
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import sys
import paras
import cPickle as pickle
import evaluation
import summarize
import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn import manifold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
sys.path.append('../tsne/bhtsne-master')
from bhtsne import *

pd = dict(paras.pd)

def load_train_test(io):
    train_db = DB(io.clean_text_file, io.dns, io.port, io.db, 'train', io.index)
    train_tweets = [tweet for tweet in train_db.get_tweets_from_db()]
    test_db = DB(io.clean_text_file, io.dns, io.port, io.db, 'test', io.index)
    test_tweets = [tweet for tweet in test_db.get_tweets_from_db()]
    return train_tweets, test_tweets


def gen_label_mapping(train_tweets, test_tweets):
    labels = set()
    for tweet in train_tweets:
        labels.add(tweet.text)
    for tweet in test_tweets:
        labels.add(tweet.text)
    labels = list(labels)
    label_map = {}
    for i, label in enumerate(labels):
        label_map[label] = i
    return label_map


def format_data_set(tweets, embedding_mode, label_map, label_instance_limit=1000000):
    features, labels = [], []
    label2num = defaultdict(int)
    # random.shuffle(tweets)
    for tweet in tweets:
        label = label_map[tweet.text]
        if label2num[label]<label_instance_limit:
            label2num[label] += 1
            features.append(gen_feature(tweet, embedding_mode))
            labels.append(label)
    # return csr_matrix(features), labels
    # print label2num
    return np.array(features), np.array(labels)


def gen_feature(tweet, embedding_model):
    spatial_feature = embedding_model.gen_spatial_feature(tweet.lat, tweet.lng)
    temporal_feature = embedding_model.gen_temporal_feature(tweet.ts)
    textual_feature = embedding_model.gen_textual_feature(tweet.words)
    # return textual_feature
    return np.concatenate([spatial_feature, temporal_feature, textual_feature])


def train(features, labels):
    # model = LogisticRegression()
    model = RandomForestClassifier()
    model.fit(features, labels)
    return model

def eval(model, features, labels):
    expected = labels
    predicted = model.predict(features)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    return accuracy_score(expected, predicted), f1_score(expected, predicted, average='micro'), f1_score(expected, predicted, average='macro'), f1_score(expected, predicted, average='weighted')  

def main(io):
    start_time = time.time()
    train_tweets, test_tweets = load_train_test(io)
    print time.time()-start_time, "loading done!"

    # embedding_model = pickle.load(open(io.models_dir+'gsm2vecPredictor.model','r'))
    best_params = summarize.get_best_params()
    for para in best_params:
        pd[para] = best_params[para]
    # pd['dim'] = 50
    embedding_model = evaluation.train(train_tweets,pd)
    print time.time()-start_time, "embedding_model training done!"

    label_map = gen_label_mapping(train_tweets, test_tweets)
    print label_map
    f_train, l_train = format_data_set(train_tweets, embedding_model, label_map)
    f_test, l_test = format_data_set(test_tweets, embedding_model, label_map)
    print time.time()-start_time, "format done!"
    # print time.time()-start_time, "format done!", f_train.shape, f_train.nnz
    # feature scaling

    standard_scaler = StandardScaler()
    f_train = standard_scaler.fit_transform(f_train)
    f_test = standard_scaler.transform(f_test)

    # for random_state in range(10):
    #     # tsne = manifold.TSNE(n_components=2, random_state=random_state, angle=0.2)
    #     # X = tsne.fit_transform(f_train)
    #     X = np.array( list( bh_tsne(f_train, theta=0.2) ) )
    #     colors = ['navy', 'turquoise', 'darkorange']
    #     plt.figure(figsize=(8,4))
    #     for i in range(len(set(l_train))):
    #         # plt.scatter(X[l_train == i, 0], X[l_train == i, 1], c=rgbs[i], s=7)
    #         plt.scatter(X[l_train == i, 0], X[l_train == i, 1], color=colors[i], s=7)
    #     plt.axis('off')
    #     # plt.savefig(io.output_dir+'tsne/'+str(theta)+'.png')
    #     plt.savefig(io.output_dir+'tsne/'+str(paras.pd['predictor'])+'-'+str(random_state)+'.pdf',bbox_inches='tight')

    model = train(f_train, l_train)
    print time.time()-start_time, "training done!"
    accuracy, micro_f1, macro_f1, weighted_f1 = eval(model, f_test, l_test)
    print time.time()-start_time, "testing done!"
    return accuracy, micro_f1, macro_f1, weighted_f1


if __name__ == '__main__':
    # para_file = sys.argv[1]
    io = IO("../run/4sq.yaml")
    print main(io)
