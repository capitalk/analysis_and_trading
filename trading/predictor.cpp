//#include <sstream> 
//#include "stdio.h" 
#include "predictor.h" 


PyObject* mk_1d_float_vec(float* raw_data, npy_intp len) { 
    return PyArray_SimpleNewFromData(1, &len, PyArray_FLOAT32, (void*) raw_data); 
}

PyObject* mk_1d_float_vec(std::vector<float>& v) { 
    float* raw_data = &(v[0]); 
    npy_intp len = v.size(); 
    return mk_1d_float_vec(raw_data, len); 
}

PyObject* mk_1d_int32_vec(int32_t* raw_data, npy_intp len) { 
    return PyArray_SimpleNewFromData(1, &len, PyArray_INT32, (void*) raw_data); 
}

PyObject* mk_1d_int32_vec(std::vector<int32_t>& v) { 
    int32_t* raw_data = &(v[0]); 
    npy_intp len = v.size(); 
    return mk_1d_int32_vec(raw_data, len); 
}

PyObject* mk_1d_uint32_vec(uint32_t* raw_data, npy_intp len) { 
    return PyArray_SimpleNewFromData(1, &len, PyArray_UINT32, (void*) raw_data); 
}

PyObject* mk_1d_uint32_vec(std::vector<uint32_t>& v) { 
    uint32_t* raw_data = &(v[0]); 
    npy_intp len = v.size(); 
    return mk_1d_uint32_vec(raw_data, len); 
}


void init_python() { 
    static bool inited = false; 
    if (!inited) { 
        inited = true;  
        Py_Initialize();
        import_array();                  
    }
}

void PythonPredictor::build_python_objects(std::string str) { 
    PyObject* py_pickle_str = PyString_FromString(str.c_str());
    
    std::cout << "Importing scikits, scipy, numpy..." << std::endl; 
    // learned model probably comes from scikits.learn 
    PyRun_SimpleString("import numpy, scipy, scikits.learn"); 
    
    std::cout << "Importing capitalk modules..." << std::endl; 
    PyRun_SimpleString("import sys"); 
    PyRun_SimpleString("sys.path.append('../analysis')"); 
    PyRun_SimpleString("sys.path.append('../trading')"); 
    PyRun_SimpleString("sys.path.append('.')"); 
    
    // need the encoder from the analysis directory 
    PyRun_SimpleString("import encoder");
    // sometimes use the dummy encoder/model from the test file 
    PyRun_SimpleString("import test_online_predictor");
    // assume the pickle string refers to online_predictor
    PyRun_SimpleString("import online_predictor");
    
    
    PyObject* cPickle = PyImport_ImportModule("cPickle");
    PyObject* loads = PyObject_GetAttrString(cPickle, "loads");
    std::cout << "Unpickling online predictor..."; 
    PyObject* py_predictor = PyObject_CallFunctionObjArgs(loads, py_pickle_str, NULL);
    Py_DECREF(py_pickle_str); 
    if (PyErr_Occurred() != NULL) { PyErr_Print(); } 
    else { std::cout << "done" <<  std::endl; }
    this->predictor = py_predictor;
    this->str_tick = PyString_FromString("tick"); 
    this->str_aggregate_frame = PyString_FromString("aggregate_frame"); 
    this->str_predict = PyString_FromString("raw_prediction");    
}

PythonPredictor::PythonPredictor(const std::istream& serialized) {
    init_python(); 
    
    // stupid shenanigans to get C++ to read an istream into a string 
    std::stringstream buffer;
    buffer << serialized.rdbuf();
    std::string contents(buffer.str());
    this->build_python_objects(contents); 
}

PythonPredictor::PythonPredictor(std::string pickle_str) { 
    init_python(); 
    this->build_python_objects(pickle_str); 
}

int PythonPredictor::predict(time_ms t) { 
    PyObject* py_t = PyLong_FromUnsignedLong(t);
    PyObject_CallMethodObjArgs(this->predictor, this->str_aggregate_frame, py_t, NULL); 
    Py_DECREF(py_t);
    if (PyErr_Occurred() != NULL) { 
        PyErr_Print(); 
        return 0; 
    } 
    
    PyObject* py_result = PyObject_CallMethodObjArgs(this->predictor, this->str_predict, NULL); 
    if (PyErr_Occurred() != NULL) { 
        PyErr_Print(); 
        return 0; 
    } 
    int result = (int) PyLong_AsLong(py_result);
    Py_DECREF(py_result); 
    return result; 
}


void PythonPredictor::update(time_ms t, fvec bids, uvec bid_sizes, fvec offers, uvec offer_sizes) {
    PyObject* py_t = PyLong_FromUnsignedLong(t);
    PyObject* py_bids = mk_1d_float_vec(bids);
    PyObject* py_bid_sizes = mk_1d_uint32_vec(bid_sizes); 
    PyObject* py_offers = mk_1d_float_vec(offers); 
    PyObject* py_offer_sizes = mk_1d_uint32_vec(offer_sizes); 
    PyObject_CallMethodObjArgs(this->predictor, this->str_tick, py_t, py_bids, py_bid_sizes, py_offers, py_offer_sizes, NULL); 
    if (PyErr_Occurred() != NULL) { PyErr_Print(); }
    // since we haven't wrapped arrays in CObject then destroying the numpy array 
    // doesn't delete the underlying data 
    Py_DECREF(py_t); 
    Py_DECREF(py_bids); 
    Py_DECREF(py_bid_sizes); 
    Py_DECREF(py_offers); 
    Py_DECREF(py_offer_sizes); 
}
