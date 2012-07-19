#!/usr/bin/env python
"""Simple file download file from s3

Usage:
s3_download_file.py -b <bucket_name> -k <key_name>  [ -f <local_filename> ] [ -d <local_directory> ]

if -f not specified will take on key_name value in current directory

"""
import os
import sys
import glob
import subprocess
import contextlib
import functools
import multiprocessing
from multiprocessing.pool import IMapIterator
from optparse import OptionParser

import boto

def get_s3_file_to_local( bucket_name, s3_key_name, s3cxn = None, local_filename = None, local_directory = None):
    if s3cxn is None:
        s3cxn = boto.connect_s3()
    if local_filename is None:
        local_filename = s3_key_name
    if local_directory is None:
        local_directory = "./"
        
    bucket = s3cxn.get_bucket(bucket_name)
    s3_key = bucket.get_key(s3_key_name)
    if s3_key is None:
        raise RuntimeError("Invalid s3 key name " + s3_key)
    else:
        s3_key.get_contents_to_filename(os.path.join(local_directory,local_filename))
    
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--bucket", dest="bucket_name")
    parser.add_option("-k", "--key_name", dest="s3_key_name")
    parser.add_option("-f", "--file", dest="local_filename", default = None)
    parser.add_option("-d", "--directory", dest="local_directory", default = None)
    (options, args) = parser.parse_args()
    assert len(args) == 0

    if options.bucket_name is None:
        print __doc__
        sys.exit()
    if options.s3_key_name is None:
        print __doc__
        sys.exit()

    get_s3_file_to_local(options.bucket_name, options.s3_key_name, None, options.local_filename, options.local_directory)
   