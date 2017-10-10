#!/bin/bash

# build sentencepiece for bpe 
sudo apt-get update
sudo apt-get install autoconf automake libtool libprotobuf9v5 protobuf-compiler libprotobuf-dev make pkg-config
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
./autogen.sh
./configure
make
make check
sudo make install
sudo ldconfig -v
cd ..

# download wmt16 data
mkdir data
cd data
wget http://www.statmt.org/europarl/v7/de-en.tgz
wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
wget http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz
wget http://data.statmt.org/wmt16/translation-task/dev.tgz
wget http://data.statmt.org/wmt16/translation-task/test.tgz
tar -xvzf de-en.tgz
tar -xvzf training-parallel-commoncrawl.tgz
tar -xvzf training-parallel-nc-v11.tgz
tar -xvzf dev.tgz
tar -xvzf test.tgz
cat europarl-v7.de-en.en commoncrawl.de-en.en training-parallel-nc-v11/news-commentary-v11.de-en.en > train.en
cat europarl-v7.de-en.de commoncrawl.de-en.de training-parallel-nc-v11/news-commentary-v11.de-en.de > train.de
git clone https://github.com/moses-smt/mosesdecoder.git
mosesdecoder/scripts/ems/support/input-from-sgm.perl < dev/newstest2014-deen-src.de.sgm > dev/newstest2014.de
mosesdecoder/scripts/ems/support/input-from-sgm.perl < dev/newstest2014-deen-ref.en.sgm > dev/newstest2014.en
mosesdecoder/scripts/ems/support/input-from-sgm.perl < dev/newstest2015-deen-src.de.sgm > dev/newstest2015.de
mosesdecoder/scripts/ems/support/input-from-sgm.perl < dev/newstest2015-deen-ref.en.sgm > dev/newstest2015.en
mosesdecoder/scripts/ems/support/input-from-sgm.perl < test/newstest2016-deen-src.de.sgm > test/newstest2016.de
mosesdecoder/scripts/ems/support/input-from-sgm.perl < test/newstest2016-deen-ref.en.sgm > test/newstest2016.en
cat dev/newstest2014.en dev/newstest2015.en > dev.en
cat dev/newstest2014.de dev/newstest2015.de > dev.de
cd ..

# bpe training and encoding
spm_train --input=data/train.en,data/train.de --model_prefix=en_de_bp3_37000 --vocab_size=37000 --model_type=bpe
spm_encode --model=en_de_bp3_37000.model --output_format=id < data/train.en > data/train.en.id
spm_encode --model=en_de_bp3_37000.model --output_format=id < data/train.de > data/train.de.id
spm_encode --model=en_de_bp3_37000.model --output_format=id < data/dev.en > data/dev.en.id
spm_encode --model=en_de_bp3_37000.model --output_format=id < data/dev.de > data/dev.de.id
spm_encode --model=en_de_bp3_37000.model --output_format=id < data/test/newstest2016.en > data/test/newstest2016.en.id
spm_encode --model=en_de_bp3_37000.model --output_format=id < data/test/newstest2016.de > data/test/newstest2016.de.id



