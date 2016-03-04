#!/bin/zsh

# parameter file
para_file='./ny.yaml'
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
  java -jar -Xmx4G $jar_file $para_file
}


# --------------------------------------------------------------------------------
# Step 3: post-processing
# --------------------------------------------------------------------------------

function post {
  python $python_dir'postprocess.py' $para_file
}

pre
# run
# post
