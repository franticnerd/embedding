gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -funroll-loops
time ./word2vec -train ./message.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 10 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 20 -min-count 1 -sentence-vectors 1
grep '_\*' vectors.txt > sentence_vectors.txt
