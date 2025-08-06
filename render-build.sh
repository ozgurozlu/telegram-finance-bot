#!/usr/bin/env bash
# TA-Lib sistem kütüphanesini yükle
apt-get update && apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install
cd ..
pip install -r requirements.txt