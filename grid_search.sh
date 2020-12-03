#!/bin/bash

python main.py --lr 0.01 --epoch 10 > test2
python main.py --lr 0.1 > test2
python main.py --lr 0.001 > test2
