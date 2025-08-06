#!/usr/bin/env bash
set -o errexit  # hata olursa dur

# Sistem paketlerini güncelle ve TA-Lib kütüphanesini yükle
apt-get update && apt-get install -y build-essential wget tar
apt-get install -y ta-lib

# Python paketlerini yükle
pip install --upgrade pip
pip install -r requirements.txt