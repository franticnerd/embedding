function package {
  rm -rf crossmap
  mkdir crossmap
  cd crossmap
  mkdir data
  mkdir code
  mkdir scripts
  mkdir bin
  cd ..
  cp -r ../data/ ./crossmap/data/
  cp -r ../code/ ./crossmap/code/
  cp -r ../scripts/ ./crossmap/scripts/
  cp ../README ./crossmap
  rm crossmap.zip
  zip -r crossmap.zip crossmap
  rm -rf ~/Downloads/crossmap/
  rm ~/Downloads/crossmap.zip
  mv crossmap ~/Downloads/
  mv crossmap.zip ~/Downloads/
}

package
