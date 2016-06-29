#!/bin/bash
# This script is based on the final round of gridsearch results that
# optimized the number of nodes in both the hidden layers.

python create_and_train.py 0 30 8
wait
python create_and_train.py 1 11 23
wait
python create_and_train.py 2 28 11
wait
python create_and_train.py 3 31 18
wait
python create_and_train.py 4 22 14
wait
python create_and_train.py 5 18 20
wait
python create_and_train.py 6 19 6
wait
python create_and_train.py 7 21 20
wait
python create_and_train.py 8 22 6
wait
python create_and_train.py 9 18 20
wait
python create_and_train.py 10 9 23
wait
python create_and_train.py 11 10 16
wait
python create_and_train.py 12 26 18
wait
python create_and_train.py 13 23 26
wait
python create_and_train.py 14 21 27
wait
python create_and_train.py 15 16 22
wait
python create_and_train.py 16 31 30
wait
python create_and_train.py 17 28 17
wait
python create_and_train.py 18 21 7
wait
python create_and_train.py 19 24 25
wait
python create_and_train.py 20 18 27
wait
python create_and_train.py 21 17 28
wait
python create_and_train.py 22 26 19
wait
python create_and_train.py 23 18 20
wait
python create_and_train.py 24 22 26
wait
python create_and_train.py 25 19 26
wait
python create_and_train.py 26 24 16
wait
python create_and_train.py 27 12 30
wait
python create_and_train.py 28 26 14
wait
python create_and_train.py 29 20 7
