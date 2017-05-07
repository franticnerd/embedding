rm -rf ~/Downloads/test-crossmap
mkdir ~/Downloads/test-crossmap
cp ~/Downloads/crossmap.zip ~/Downloads/test-crossmap
cd ~/Downloads/test-crossmap
unzip crossmap.zip
cd ./crossmap/scripts
./toy.sh
./ny.sh
./la.sh

