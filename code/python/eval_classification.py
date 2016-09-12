from zutils.twitter.tweet_database import TweetDatabase as DB
from io_utils import IO
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import sys


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


def format_data_set(tweets, embedding_mode, label_map):
    features, labels = [], []
    for tweet in tweets:
        features.append(gen_feature(tweet, embedding_mode))
        labels.append(label_map[tweet.text])
    return features, labels


def gen_feature(tweet, embedding_model):
    def gen_spatial_feature(lat, lng):
        return np.random.rand(100)
    def gen_temporal_feature(ts):
        return np.random.rand(100)
    def gen_textual_feature(words):
        return np.random.rand(100)
    spatial_feature = gen_spatial_feature(tweet.lat, tweet.lng)
    temporal_feature = gen_temporal_feature(tweet.ts)
    textual_feature = gen_textual_feature(tweet.words)
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


def main(io, embedding_model):
    train_tweets, test_tweets = load_train_test(io)
    label_map = gen_label_mapping(train_tweets, test_tweets)
    print label_map
    f_train, l_train = format_data_set(train_tweets, embedding_model, label_map)
    f_test, l_test = format_data_set(test_tweets, embedding_model, label_map)
    model = train(f_train, l_train)
    accuracy = eval(model, f_test, l_test)
    return accuracy


if __name__ == '__main__':
    para_file = sys.argv[1]
    io = IO(para_file)
    main(io, None)
