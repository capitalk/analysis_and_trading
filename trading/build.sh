#!/bin/bash
g++ predictor.cpp test_predictor.cpp  -I. -I/usr/local/lib/python2.7/site-packages/numpy/core/include -I/usr/local/include/python2.7 -L/usr/local/lib/ -lpython2.7 -pthread -lutil -ldl -o test_predictor
