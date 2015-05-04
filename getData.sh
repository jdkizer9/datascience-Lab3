#!/bin/bash
mkdir data
wget -P data/ http://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip -d data/ data/ml-10m.zip