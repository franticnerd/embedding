#!/bin/zsh
# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8
# source /Users/keyangzhang/Documents/UIUC/Engineer/Python/venv/bin/activate

# parameter file
para_file='./4sq.yaml'
python_dir='../python/'

# --------------------------------------------------------------------------------
# Step 1: preprocessing.
# --------------------------------------------------------------------------------

function pre {
  python $python_dir'preprocess.py' $para_file
}

# --------------------------------------------------------------------------------
# Step 2: run the algorithms.
# --------------------------------------------------------------------------------
function run {
  time ../doc2vec/word2vec -train ../../data/la/input/message.txt -output ../../data/la/output/vectors.txt -cbow 0 -size 100 -window 10 -negative 10 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
  grep '_\*' ../../data/la/output/vectors.txt > ../../data/la/output/embed.txt
}

# --------------------------------------------------------------------------------
# Step 3: post-processing
# --------------------------------------------------------------------------------

function post {
  # python $python_dir'postprocess.py' $para_file
  python $python_dir'eval_classification.py' $para_file
}

# pre
# run
post
