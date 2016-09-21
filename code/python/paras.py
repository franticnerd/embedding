from gsm2vec import *

def convert_ts(ts):
	# return (ts/60)%(60*24*7)
	# return (ts/3600)%(24*7)
	# return (ts/3600)%24
	# return ts/3600
	return (ts)%(3600*24*7)

def kernel(u, h=1):
	u /= h
	return math.e**(-u*u/2)

# pd stands for parasDict
pd = dict()
pd['dim'] = 100
pd['negative'] = 1
pd['alpha'] = 0.02
pd['adaptive_lr'] = 1
pd['samples'] = 10
pd['threads'] = 10
pd['ns_refuse_percent'] = 0
pd['ns_candidate_num'] = 1

pd['dataset'] = 'la'
# pd['dataset'] = '4sq'
pd['train_ratio'] = 0.8
pd['data_size'] = 1000000
pd['fake_num'] = 10
pd['predict_type'] = 'w'
pd['rand_seed'] = 2

pd['ntList'] = ['l','t','w']
# pd['etList'] = ['ll','tt','ww','lt','tw','wl']
# pd['etList'] = ['ww','lt','tw','wl']
# pd['etList'] = ['ll','tt','ww','lt','tw','wl','tl','wt','lw']
pd['etList'] = ['ww','lt','tw','wl','tl','wt','lw']
# pd['etList'] = ['lt','tw','wl']
# pd['etList'] = ['wl']
pd['second_order'] = 0
pd['use_context_vec'] = 1
pd['version'] = 0

pd['predictor'] = Gsm2vecPredictor
# pd['predictor'] = PmiPredictor
# pd['predictor'] = SvdPredictor
# pd['predictor'] = TfidfPredictor
pd['gsm2vec'] = Gsm2vec_line
# pd['gsm2vec'] = Gsm2vec
# pd['gsm2vec'] = Gsm2vec_relation
pd['lClus'] = LMeanshiftClus
# pd['tClus'] = TVoidClus
pd['tClus'] = TMeanshiftClus

pd['kernel_candidate_num'] = 0
pd['bandwidth_l'] = 0.01
pd['bandwidth_t'] = 1000
pd['kernel_bandwidth_l'] = 0.01
pd['kernel_bandwidth_t'] = 1000
pd['nb_num'] = 20
pd['grid_num'] = 20

pd['convert_ts'] = convert_ts
pd['kernel'] = kernel
pd['line_dir'] = '../python/line-package-second/'


def pd2string(pd):
	s = ''
	s += ', '.join([para+': '+str(pd[para]) for para in ['dataset','train_ratio','data_size','fake_num','predict_type','rand_seed']])+'\n'
	s += ', '.join([para+': '+str(pd[para]) for para in ['ntList','etList','second_order','use_context_vec','version']])+'\n'
	s += ', '.join([para+': '+str(pd[para]) for para in ['predictor','gsm2vec','lClus','tClus']])+'\n'
	s += ', '.join([para+': '+str(pd[para]) for para in ['dim','negative','alpha','samples','threads','adaptive_lr','ns_refuse_percent','ns_candidate_num']])+'\n'
	s += ', '.join([para+': '+str(pd[para]) for para in ['kernel_candidate_num','bandwidth_l','bandwidth_t','kernel_bandwidth_l','kernel_bandwidth_t','nb_num','grid_num']])+'\n'
	return s
