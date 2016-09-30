from zutils.twitter.tweet_database import TweetDatabase as DB
from io_utils import IO
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import paras
import cPickle as pickle
<<<<<<< HEAD
import evaluation
import summarize
import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

pd = dict(paras.pd)

def load_train_test(io):
    train_db = DB(io.clean_text_file, io.dns, io.port, io.db, 'train', io.index)
    train_tweets = [tweet for tweet in train_db.get_tweets_from_db()]
    test_db = DB(io.clean_text_file, io.dns, io.port, io.db, 'test', io.index)
    test_tweets = [tweet for tweet in test_db.get_tweets_from_db()]
    return train_tweets[:50000], test_tweets


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


def format_data_set(tweets, embedding_mode, label_map):
    features, labels = [], []
    for tweet in tweets:
        features.append(gen_feature(tweet, embedding_mode))
        labels.append(label_map[tweet.text])
    return csr_matrix(features), labels
    # return features, labels


def gen_feature(tweet, embedding_model):
    spatial_feature = embedding_model.gen_spatial_feature(tweet.lat, tweet.lng)
    temporal_feature = embedding_model.gen_temporal_feature(tweet.ts)
    textual_feature = embedding_model.gen_textual_feature(tweet.words)
    return np.concatenate([spatial_feature, temporal_feature, textual_feature])


def train(features, labels):
    model = LogisticRegression()
    model.fit(features, labels)
    return model

def eval(model, features, labels):
    expected = labels
    predicted = model.predict(features)
    print(metrics.classification_report(expected, predicted))
    return accuracy_score(expected, predicted)

def main(io):
    start_time = time.time()
    train_tweets, test_tweets = load_train_test(io)
    print time.time()-start_time, "loading done!"

    # embedding_model = pickle.load(open(io.models_dir+'gsm2vecPredictor.model','r'))
    best_params = summarize.get_best_params()
    for para in best_params:
        pd[para] = best_params[para]
    embedding_model = evaluation.train(train_tweets,pd)
    print time.time()-start_time, "embedding_model training done!"

    label_map = gen_label_mapping(train_tweets, test_tweets)
    print label_map
    f_train, l_train = format_data_set(train_tweets, embedding_model, label_map)
    f_test, l_test = format_data_set(test_tweets, embedding_model, label_map)
    # print time.time()-start_time, "format done!"
    print time.time()-start_time, "format done!", f_train.shape, f_train.nnz
    # feature scaling
    standard_scaler = StandardScaler()
    f_train = standard_scaler.fit_transform(f_train)
    f_test = standard_scaler.transform(f_test)
    model = train(f_train, l_train)
    print time.time()-start_time, "training done!"
    accuracy = eval(model, f_test, l_test)
    print time.time()-start_time, "testing done!"
    return accuracy


if __name__ == '__main__':
    # para_file = sys.argv[1]
    io = IO("../run/4sq.yaml")
    print main(io)
