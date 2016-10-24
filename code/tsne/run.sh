#!/bin/sh

vector_file=$1
output_file=$2

./fmt-in -vector ${vector_file} -label label.txt -data bhtsne-master/data.dat -nshows 0
cd bhtsne-master
./bh_tsne
cd ..
./fmt-out -vector bhtsne-master/result.dat -label label.txt -data ${output_file}