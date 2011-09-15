#include "stdio.h" 
#include <iostream>
using namespace std;

#include "Python.h"
#include "numpy/arrayobject.h"

void init(int argc, char* argv[]) { 
    Py_Initialize();
    PySys_SetArgvEx(argc, argv, 0); 
    import_array();    
}

PyObject* mk_1d_float_vec(float* raw_data, npy_intp len) { 
    return PyArray_SimpleNewFromData(1, &len, PyArray_FLOAT32, (void*) raw_data); 
}

int main(int argc, char *argv[]) {
    init(argc, argv); 
    PySys_SetPath("."); 
    PyRun_SimpleString("import online_aggregator");
    PyRun_SimpleString("import online_features");
    PyRun_SimpleString("agg = online_aggregator.OnlineAggregator()");
    PyRun_SimpleString("agg.add_feature('bid', online_features.best_bid)");
    PyRun_SimpleString("agg.add_feature('offer', online_features.best_offer)");

    printf("Ran startup code\n"); 
    PyObject *globals = PyImport_AddModule("__main__");
    printf("Got globals\n"); 
    PyObject *agg = PyObject_GetAttrString(globals,"agg");
    printf("Getting attr\n"); 
    PyObject* millisecond_update = PyString_FromString("millisecond_update"); 

    npy_intp len = 4;
    float raw_data[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    printf("Creating numpy array\n"); 
    PyObject* data_arg = mk_1d_float_vec(raw_data, len);    
    Py_INCREF(data_arg);
    Py_INCREF(data_arg);
    Py_INCREF(data_arg);
    Py_INCREF(data_arg);
    Py_INCREF(data_arg);
    Py_INCREF(data_arg);
    Py_INCREF(data_arg);
    Py_INCREF(data_arg);
    Py_INCREF(data_arg);
    
    
    printf("Created\n"); 
    PyObject_Print(data_arg, stdout, 0);
    PyObject* time_arg300 =  PyInt_FromLong(300L);
    PyObject* time_arg310 =  PyInt_FromLong(310L);
    printf("[C++] First call\n"); 
    
    PyObject_CallMethodObjArgs(agg, millisecond_update, time_arg300, data_arg, data_arg, data_arg, data_arg, NULL); 
    if (PyErr_Occurred() != NULL) { PyErr_Print(); } 
    PyObject_Print(data_arg, stdout, 0);
    
    printf("[C++] Second call\n"); 
    PyObject_CallMethodObjArgs(agg, millisecond_update, time_arg310, data_arg, data_arg, data_arg, data_arg, NULL); 
    if (PyErr_Occurred() != NULL) { PyErr_Print(); } 
    
    PyObject_Print(data_arg, stdout, 0);
    PyObject* time_arg320 =  PyInt_FromLong(320L);
    PyObject* aggregate_frame = PyString_FromString("aggregate_frame"); 
    
    printf("[C++] Calling aggregate frame\n"); 
    PyObject_CallMethodObjArgs(agg, aggregate_frame, time_arg320, NULL); 
    if (PyErr_Occurred() != NULL) { PyErr_Print(); } 
    printf("\n"); 
    Py_DECREF(data_arg);
    return 0; 
}

