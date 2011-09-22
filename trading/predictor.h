#include <iostream>
#include <string> 
#include <vector> 
#include <sstream> 
#include <stdio.h> 
#include "Python.h"
#include "numpy/arrayobject.h"

#ifndef PREDICTOR_H__ 
#define PREDICTOR_H__ 

typedef unsigned long time_ms; 
typedef std::vector<float>& fvec; 
typedef std::vector<int32_t>& ivec; 
typedef std::vector<uint32_t>& uvec; 

class PythonPredictor { 
private: 
    PyObject* predictor;
    PyObject* str_tick; 
    PyObject* str_aggregate_frame; 
    PyObject* str_predict; 
    
    void build_python_objects(std::string str); 
    
    // can't construct without getting a python object description from 
    // a pickle string 
    PythonPredictor(); 
public: 
    PythonPredictor(const std::istream&); 
    PythonPredictor(std::string); 
    int predict(time_ms t); 
    void update(time_ms t, fvec bids, uvec bid_sizes, fvec offers, uvec offer_sizes); 
};
#endif //  PREDICTOR_H__ 
