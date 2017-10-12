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

# download data
mkdir translation-data
cd translation-data
wget http://www.statmt.org/europarl/v7/de-en.tgz
wget http://data.statmt.org/wmt17/translation-task/rapid2016.tgz
wget http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz
wget http://data.statmt.org/wmt16/translation-task/dev.tgz
wget http://data.statmt.org/wmt16/translation-task/test.tgz
tar -xvzf de-en.tgz
tar -xvzf rapid2016.tgz
tar -xvzf training-parallel-nc-v12.tgz
tar -xvzf dev.tgz
tar -xvzf test.tgz
cat europarl-v7.de-en.en rapid2016.de-en.en training/news-commentary-v12.de-en.en > train.en
cat europarl-v7.de-en.de rapid2016.de-en.de training/news-commentary-v12.de-en.de > train.de
git clone https://github.com/moses-smt/mosesdecoder.git
mosesdecoder/scripts/ems/support/input-from-sgm.perl < dev/newstest2014-deen-src.de.sgm > dev/newstest2014.de
mosesdecoder/scripts/ems/support/input-from-sgm.perl < dev/newstest2014-deen-ref.en.sgm > dev/newstest2014.en
mosesdecoder/scripts/ems/support/input-from-sgm.perl < dev/newstest2015-deen-src.de.sgm > dev/newstest2015.de
mosesdecoder/scripts/ems/support/input-from-sgm.perl < dev/newstest2015-deen-ref.en.sgm > dev/newstest2015.en
mosesdecoder/scripts/ems/support/input-from-sgm.perl < test/newstest2016-deen-src.de.sgm > test/newstest2016.de
mosesdecoder/scripts/ems/support/input-from-sgm.perl < test/newstest2016-deen-ref.en.sgm > test/newstest2016.en
cat dev/newstest2014.en dev/newstest2015.en > dev.en
cat dev/newstest2014.de dev/newstest2015.de > dev.de

# bpe training and encoding
spm_train --input=train.en,train.de --model_prefix=de-en_bp3_32000 --vocab_size=32000 --model_type=bpe
spm_encode --model=de-en_bp3_32000.model --output_format=id < train.de > train.de.id
spm_encode --model=de-en_bp3_32000.model --output_format=id --extra_options=bos:eos < train.en > train.en.id
spm_encode --model=de-en_bp3_32000.model --output_format=id < dev.de > dev.de.id
spm_encode --model=de-en_bp3_32000.model --output_format=id --extra_options=bos:eos < dev.en > dev.en.id
spm_encode --model=de-en_bp3_32000.model --output_format=id < test/newstest2016.de > test/newstest2016.de.id
spm_encode --model=de-en_bp3_32000.model --output_format=id --extra_options=bos:eos < test/newstest2016.en > test/newstest2016.en.id



