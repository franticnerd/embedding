time ./word2vec -train ../../data/la/input/message.txt -output ../../data/la/output/vectors.txt -cbow 0 -size 100 -window 10 -negative 10 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 20 -min-count 1 -sentence-vectors 1
# grep '_\*' ../../data/la/output/vectors.txt > ../../data/la/output/embed.txt
