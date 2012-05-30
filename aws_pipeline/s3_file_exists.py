#!/usr/bin/env python
"""
Usage:
s3_file_exists.py <file> <bucket> 

"""
import os
import sys
import glob
import subprocess
import errno
import fnmatch
import commands
from optparse import OptionParser
import boto
import mics


def check_s3_bucket_exists(s3cxn, bucket_name):
    if s3cxn is None:
        print "No connection to s3"
    bucket = s3cxn.get_bucket(bucket_name)
    if bucket is None:
        return False
    else:
        return bucket

def check_for_file_in_bucket(filename, bucket_name):
    if s3cxn is None:
        print "No connection to s3"

    bucket = check_s3_bucket_exists(s3cxn, bucket_name)    
    if bucket is False:
            print "No such bucket"
            exit;
            
    print "Checking bucket %s" % bucket_name
    bucket = s3cxn.get_bucket(bucket_name)

    keys = bucket.get_all_keys()
    if len(keys) == 0:
        print "There are no keys in %s bucket" % bucket_name
        exit;
        
    print "Examining %d keys" % len(keys),

    found_bytes = 0

    names = [fn.name  for fn in keys]
    match_names = fnmatch.filter(names, fn_regex);
    print "of which %d match regex" %  len(match_names)

    for n in match_names:
        key = bucket.get_key(n)
        found_bytes += key.size

    print "Found %d files in bucket %s with %d bytes (%d kB) (%d MB) (%d GB)" % (len(match_names), bucket_name, found_bytes, found_bytes/1000, found_bytes/1000000, found_bytes/1000000000)
       
    return (bucket_name, len(match_names), found_bytes)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--all-buckets", action='store_true', dest="all_buckets", help="Look for file in all buckets", default=False)

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print __doc__
        sys.exit()
        exit(-1)

    s3cxn = boto.connect_s3()
    print "Args: ", args

    fn_regex = args[0]
    if len(args) == 2:
        bucket_name = args[1]
    else:
        bucket_name = None
    if bucket_name is None:
        print "Checking all buckets"
        buckets = s3cxn.get_all_buckets()
        for b in buckets:
            (bucket_name, num_files_found, bytes_found) = check_for_file_in_bucket(fn_regex, b)
    else:
        (bucket_name, num_files_found, bytes_found) = check_for_file_in_bucket(fn_regex, bucket_name)

    
