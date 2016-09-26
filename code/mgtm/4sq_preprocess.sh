
dns='dmserv4.cs.illinois.edu'
port=11111
db='foursquare'
col='train'
data_dir='/Users/chao/Dropbox/Research/embedding/data/4sq/mgtm/'
python preprocess_mgtm.py $dns $port $db $col $data_dir
