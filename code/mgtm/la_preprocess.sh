
dns='dmserv4.cs.illinois.edu'
port=11111
db='tweet-la'
col='raw'
data_dir='/Users/chao/Dropbox/Research/embedding/data/la/mgtm/'
tweet_id_file='/Users/chao/Dropbox/Research/embedding/data/la/training_tweet_ids'
python preprocess_mgtm.py $dns $port $db $col $data_dir $tweet_id_file
