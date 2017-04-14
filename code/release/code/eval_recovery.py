from paras import load_params
from dataset import read_tweets, get_voca
from embed import *
from evaluator import QuantitativeEvaluator


def set_rand_seed(pd):
    rand_seed = pd['rand_seed']
    np.random.seed(rand_seed)
    random.seed(rand_seed)


def read_data(pd):
    start_time = time.time()
    tweets = read_tweets(pd['tweet_file'])
    random.shuffle(tweets)
    voca = get_voca(tweets, pd['voca_min'], pd['voca_max'])
    train_data, test_data = tweets[:-pd['test_size']], tweets[-pd['test_size']:]
    print 'Reading data done, elapsed time: ', round(time.time()-start_time)
    print 'Total number of tweets: ', len(tweets)
    print 'Number of training tweets: ', len(train_data)
    print 'Number of test tweets: ', len(test_data)
    return train_data, test_data, voca


def train_model(train_data, voca):
    start_time = time.time()
    predictor = EmbedPredictor(pd)
    predictor.fit(train_data, voca)
    print 'Model training done, elapsed time: ', round(time.time()-start_time)
    return predictor


def predict(model, test_data, pd):
    start_time = time.time()
    for t in pd['predict_type']:
        evaluator = QuantitativeEvaluator(predict_type=t)
        evaluator.get_ranks(test_data, model)
        mrr, mr = evaluator.compute_mrr()
        print 'Type: ', evaluator.predict_type, 'mr:', mr, 'mrr:', mrr
    print 'Prediction done. Elapsed time: ', round(time.time()-start_time)


def write_model(model, pd):
    directory = pd['model_dir']
    if not os.path.isdir(directory):
        os.makedirs(directory)
    for nt, vecs in model.nt2vecs.items():
        with open(directory+nt+'.txt', 'w') as f:
            for node, vec in vecs.items():
                if nt=='l':
                    node = model.lClus.centroids[node]
                l = [str(e) for e in [node, list(vec)]]
                f.write('\x01'.join(l)+'\n')

def run(pd):
    set_rand_seed(pd)
    train_data, test_data, voca = read_data(pd)
    model = train_model(train_data, voca)
    predict(model, test_data, pd)
    write_model(model, pd)

if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    run(pd)
