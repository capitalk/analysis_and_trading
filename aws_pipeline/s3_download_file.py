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

def main(bucket_name, s3_key_name, local_filename, local_directory):
    s3cxn = boto.connect_s3()
    get_s3_file_to_local(s3cxn, bucket_name, s3_key_name, local_filename, local_directory)
    


def get_s3_file_to_local(s3cxn, bucket_name, s3_key_name, local_filename, local_directory):
    if s3cxn is None:
        raise RuntimeError("Invalid s3 connection")
    if local_filename is None:
        local_filename = s3_key_name
    bucket = s3cxn.get_bucket(bucket_name)
    s3_key = bucket.get_key(s3_key_name)
    if s3_key is None:
        raise RuntimeError("Invalid s3 key name")
    else:
        s3_key.get_contents_to_filename(local_directory+local_filename)
    
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--bucket", dest="bucket_name")
    parser.add_option("-k", "--key_name", dest="s3_key_name")
    parser.add_option("-f", "--file", dest="local_filename")
    parser.add_option("-d", "--directory", dest="local_directory")
    (options, args) = parser.parse_args()
    #print options
    if options.local_filename is None:
        options.local_filename = options.s3_key_name
    if options.bucket_name is None:
        print __doc__
        sys.exit()
    if options.s3_key_name is None:
        print __doc__
        sys.exit()
    if options.local_directory is None:
        options.local_directory = "./"

    if options.local_directory[-1] != '/':
        print __doc__
        print "NOTE: Directory name should end with /" 
        sys.exit()
    #print args
    #if len(args) < 3:
    #    print __doc__
    #    sys.exit()
    kwargs = dict(bucket_name=options.bucket_name, s3_key_name=options.s3_key_name, local_filename=options.local_filename, local_directory=options.local_directory)
    main(*args, **kwargs)
