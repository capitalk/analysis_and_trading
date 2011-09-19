#!/usr/bin/env python



 # file exists and 'finished' flag is true 

import os
import sys
import h5py

def hdf_complete(filename):
    if not os.path.exists(filename): return False 
    try:
        f = h5py.File(filename, 'r')
        finished = 'finished' in f.attrs and f.attrs['finished']
        f.close() 
        return finished
    except:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: testHdfFinished.py <filename>"
        sys.exit()
    else:
        ret = hdf_complete(sys.argv[1])
        if ret == True:
            print sys.argv[1], " complete"
            sys.exit(0)
        else:
            print sys.argv[1], " NOT complete"
            sys.exit(1)
        
 
