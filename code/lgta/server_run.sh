
input_dir='/home/czhang82/exp/proj/embedding/data/4sq/lgta/input/'
output_dir='/home/czhang82/exp/proj/embedding/data/4sq/lgta/output/'
n_region=100
n_topic=50

cd main
matlab -r demo($input_dir, $output_dir, $n_region, $n_topic)
cd ..
