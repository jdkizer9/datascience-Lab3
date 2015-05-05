#!/bin/bash
mkdir data
wget -P data/ http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip -d data/ data/ml-20m.zip