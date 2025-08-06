#!/usr/bin/env bash
set -o errexit

apt-get update && apt-get install -y build-essential wget tar
apt-get install -y ta-lib

pip install --upgrade pip
pip install -r requirements.txt