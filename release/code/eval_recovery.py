from paras import load_params
from dataset import read_tweets, get_voca
from embed import *
from evaluator import QuantitativeEvaluator, QualitativeEvaluator


def set_rand_seed(pd):
    rand_seed = pd['rand_seed']
    np.random.seed(rand_seed)
    random.seed(rand_seed)


def read_data(pd):
    start_time = time.time()
    tweets = read_tweets(pd['tweet_file'])
    random.shuffle(tweets)
    voca = get_voca(tweets, pd['voca_min'], pd['voca_max'])
    train_data, test_data = tweets[pd['test_size']:], tweets[:pd['test_size']]
    print 'Reading data done, elapsed time: ', round(time.time()-start_time)
    print 'Total number of tweets: ', len(tweets)
    print 'Number of training tweets: ', len(train_data)
    print 'Number of test tweets: ', len(test_data)
    return train_data, test_data, voca


def train_model(train_data, voca):
    start_time = time.time()
    predictor = CrossMap(pd)
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
                if nt=='t':
                    node = model.tClus.centroids[node]
                l = [str(e) for e in [node, list(vec)]]
                f.write('\x01'.join(l)+'\n')

def run_case_study(model, pd):
    start_time = time.time()
    evaluator = QualitativeEvaluator(model, pd['case_dir'])
    for word in ['food', 'restaurant', 'beach', 'weather', 'clothes', 'nba']:
        evaluator.getNbs1(word)
    for location in [[34.043021,-118.2690243], [33.9424, -118.4137], [34.008, -118.4961], [34.0711, -118.4434]]:
        evaluator.getNbs1(location)
    evaluator.getNbs2('outdoor', 'weekend')
    print 'Case study done. Elapsed time: ', round(time.time()-start_time)

def run(pd):
    set_rand_seed(pd)
    train_data, test_data, voca = read_data(pd)
    model = train_model(train_data, voca)
    predict(model, test_data, pd)
    write_model(model, pd)
    run_case_study(model, pd)

if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    run(pd)
