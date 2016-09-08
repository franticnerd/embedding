from gsm2vec import *

def convert_ts(ts):
	# return (ts/60)%(60*24*7)
	# return (ts/3600)%(24*7)
	# return (ts/3600)%24
	# return ts/3600
	return (ts)%(3600*24*7)

# pd stands for parasDict
pd = dict()
pd["dim"] = 100
pd["negative"] = 1
pd["alpha"] = 0.025
# pd["alpha"] = 0.01
pd["adaptive_lr"] = 1
pd["samples"] = 10
pd["threads"] = 10
pd["ns_refuse_percent"] = 0
pd["ns_candidate_num"] = 1

pd["train_ratio"] = 0.98
pd["data_size"] = 100000
pd["fake_num"] = 10
pd["predict_type"] = 'w'
pd["rand_seed"] = 2

pd["ntList"] = ['l','t','w']
# pd["etList"] = ['ll','tt','ww','lt','tw','wl']
pd["etList"] = ['ww','lt','tw','wl']
# pd["etList"] = ['ll','tt','ww','lt','tw','wl','tl','wt','lw']
# pd["etList"] = ['ww','lt','tw','wl','tl','wt','lw']
# pd["etList"] = ['lt','tw','wl']
# pd["etList"] = ['wl']
pd["second_order"] = 0
pd["use_context_vec"] = 1

pd["gsm2vec"] = Gsm2vec_line
# pd["gsm2vec"] = Gsm2vec
# pd["gsm2vec"] = Gsm2vec_relation
pd["lClus"] = MeanshiftClus
# pd["tClus"] = TVoidClus
pd["tClus"] = MeanshiftClus

pd["kernel_candidate_num"] = 1
pd["bandwidth_kernel"] = 1
pd["bandwidth_l"] = 0.01
pd["bandwidth_t"] = 1000
pd["nb_num"] = 20
pd["grid_num"] = 20
pd["convert_ts"] = convert_ts

def pd2string(pd):
	s = ""
	s += ", ".join([para+": "+str(pd[para]) for para in ["train_ratio","data_size","fake_num","predict_type","rand_seed"]])+"\n"
	s += ", ".join([para+": "+str(pd[para]) for para in ["ntList","etList","second_order"]])+"\n"
	s += ", ".join([para+": "+str(pd[para]) for para in ["gsm2vec","lClus","tClus"]])+"\n"
	s += ", ".join([para+": "+str(pd[para]) for para in ["dim","negative","alpha","samples","threads","adaptive_lr","ns_refuse_percent","ns_candidate_num"]])+"\n"
	s += ", ".join([para+": "+str(pd[para]) for para in ["bandwidth_kernel","bandwidth_l","bandwidth_t","nb_num","grid_num"]])+"\n"
	return s
