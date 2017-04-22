from param_handler import yaml_loader
from embed import *

def load_params(para_file):
    if para_file is None:
        para = set_default_params()
    else:
        para = yaml_loader().load(para_file)
    para['embed_algo'] = GraphEmbedLine
    para['rand_seed'] = 1
    para['category_list'] = []
    return para


def set_default_params():
    pd = dict()
    pd['data_dir'] = '/Users/chao/data/projects/embedding/tweets-dev/'
    pd['tweet_file'] = pd['data_dir'] + 'input/tweets.txt'
    pd['result_dir'] = pd['data_dir'] + 'output/'
    pd['model_dir'] = pd['data_dir'] + 'model/'

    pd['voca_min'] = 0
    pd['voca_max'] = 20000
    pd['dim'] = 10
    pd['negative'] = 1
    pd['alpha'] = 0.02 # learning rate
    pd['epoch'] = 1
    pd['nt_list'] = ['w', 'l', 't']
    pd['predict_type'] = ['w', 'l', 't']
    pd['test_size'] = 100
    pd['kernel_nb_num'] = 1 # used for efficiency reason (requested by fast k-nearest-neighbor search)
    pd['bandwidth_l'] = 0.002 # used only in LClus, should be of similar magnitude as grid_len
    pd['bandwidth_t'] = 1000.0 # used only in LClus
    pd['kernel_bandwidth_l'] = 0.005
    pd['kernel_bandwidth_t'] = 4000.0
    pd['second_order'] = 1
    pd['use_context_vec'] = 1 # used only in second order graph embedding, suggest value: 1
    return pd

