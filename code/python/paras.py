from gsm2vec import *

# pd stands for parameter dictionary
pd = dict()
pd['dim'] = 100
pd['negative'] = 1
pd['alpha'] = 0.02
pd['adaptive_lr'] = 1
pd['samples'] = 1
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
pd['etList'] = ['ll','tt','ww','lt','tw','wl','tl','wt','lw']
# pd['etList'] = ['ww','lt','tw','wl','tl','wt','lw']
# pd['etList'] = ['lt','tw','wl']
# pd['etList'] = ['wl']
pd['second_order'] = 0
pd['use_context_vec'] = 1
pd['version'] = 0

pd['predictor'] = TensorPredictor
# pd['predictor'] = Gsm2vecPredictor
# pd['predictor'] = PmiPredictor
# pd['predictor'] = SvdPredictor
# pd['predictor'] = TfidfPredictor
# pd['gsm2vec'] = Gsm2vec_line
# pd['gsm2vec'] = Gsm2vec
pd['gsm2vec'] = Gsm2vec_relation
pd['lClus'] = LMeanshiftClus
# pd['tClus'] = TVoidClus
pd['tClus'] = TMeanshiftClus

pd['kernel_nb_num_l'] = 10
pd['kernel_nb_num_t'] = 10
pd['bandwidth_l'] = 0.01
pd['bandwidth_t'] = 1000.0
pd['kernel_bandwidth_l'] = 0.01
pd['kernel_bandwidth_t'] = 1000.0
pd['grid_num'] = 20

pd['tensor_rank'] = 10

pd['line_dir'] = '../line_gsm2vec/'
pd['job_id'] = '0'


def pd2string(pd):
	s = ''
	s += ', '.join([para+': '+str(pd[para]) for para in ['dataset','train_ratio','data_size','fake_num','predict_type','rand_seed']])+'\n'
	s += ', '.join([para+': '+str(pd[para]) for para in ['ntList','etList','second_order','use_context_vec','version']])+'\n'
	s += ', '.join([para+': '+str(pd[para]) for para in ['predictor','gsm2vec','lClus','tClus']])+'\n'
	s += ', '.join([para+': '+str(pd[para]) for para in ['dim','negative','alpha','samples','threads','adaptive_lr','ns_refuse_percent','ns_candidate_num']])+'\n'
	s += ', '.join([para+': '+str(pd[para]) for para in ['kernel_nb_num_l','kernel_nb_num_t','bandwidth_l','bandwidth_t','kernel_bandwidth_l','kernel_bandwidth_t','grid_num']])+'\n'
	return s
