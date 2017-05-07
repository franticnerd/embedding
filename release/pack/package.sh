rm -rf crossmap
mkdir crossmap
cd crossmap
mkdir data
mkdir code
mkdir scripts

echo 'Start copying data!'
cd ..
cp -r ../data/ ./crossmap/
cp -r ../code/ ./crossmap/
cp -r ../scripts/ ./crossmap/
cp ../README ./crossmap
echo 'Finished copying data!'

rm crossmap.zip
zip -r crossmap.zip crossmap
echo 'Finished zipping files!'

rm -rf ~/Downloads/crossmap/
rm -f ~/Downloads/crossmap.zip
mv crossmap ~/Downloads/
mv crossmap.zip ~/Downloads/
