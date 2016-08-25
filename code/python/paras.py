from gsm2vec import *

def convert_ts(ts):
	return (ts/3600)%(24*7)
	# return (ts/3600)%24

# pd stands for parasDict
pd = dict()
pd["dim"] = 100
pd["negative"] = 1
pd["alpha"] = 0.025
pd["samples"] = 10
pd["threads"] = 10

pd["train_ratio"] = 0.98
pd["data_size"] = 100000
pd["fake_num"] = 10
pd["predict_type"] = 't'
pd["rand_seed"] = 2

pd["ntList"] = ['l','t','w']
# pd["etList"] = ['ll','tt','ww','lt','tw','wl']
# pd["etList"] = ['ww','lt','tw','wl']
pd["etList"] = ['ll','tt','ww','lt','tw','wl','tl','wt','lw']
# pd["etList"] = ['ww','lt','tw','wl','tl','wt','lw']
# pd["etList"] = ['lt','tw','wl']

pd["lClus"] = MeanshiftClus
pd["tClus"] = TVoidClus
# pd["tClus"] = MeanshiftClus
pd["convert_ts"] = convert_ts
pd["bandwidth_l"] = 0.01
pd["bandwidth_t"] = 10
pd["nb_num"] = 20
pd["grid_num"] = 20

def pd2string(pd):
	s = ""
	s += ", ".join([para+": "+str(pd[para]) for para in ["dim","negative","alpha","samples","threads"]])+"\n"
	s += ", ".join([para+": "+str(pd[para]) for para in ["train_ratio","data_size","fake_num","predict_type","rand_seed"]])+"\n"
	s += ", ".join([para+": "+str(pd[para]) for para in ["ntList","etList"]])+"\n"
	s += ", ".join([para+": "+str(pd[para]) for para in ["lClus","tClus","convert_ts","bandwidth_l","bandwidth_t","nb_num","grid_num"]])+"\n"
	return s
