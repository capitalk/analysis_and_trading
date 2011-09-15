#!/usr/bin/env python
"""Move a set of ticks in csv.gz format to s3. Can take directory or file

Usage:
s3_move_ticks.py <directory_or_file> to move 

"""
import os
import sys
import glob
import subprocess
import errno
import commands
from optparse import OptionParser
import s3_multipart_upload
import boto
import mics
from mics import MIC_LIST


BUCKET_PREFIX="capk-"

def check_s3_bucket_exists(s3cxn, bucket_name):
    if s3cxn is None:
        print "No connection to s3"
    bucket = s3cxn.get_bucket(bucket_name)
    if bucket is None:
        return False
    else:
        return bucket


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--create-buckets", dest="create_buckets", help="Create buckets on s3 if they don't exist now", default=False)
    (options, args) = parser.parse_args()
    if len(args) < 1:
        print __doc__
        sys.exit()
    kwargs = dict(mode="test")
    print args
    path = args[0]
    files = []
    if not os.path.exists(path):
        print "Specified path does not exist: ", path 
        sys.exit(0)
    if os.path.isdir(path):
        if path[-1] != os.path.sep:
            path=path+os.path.sep
        files = glob.glob(path+"*.csv.gz")
        mic = path.split(os.path.sep)[-3]
        if mic not in MIC_LIST:
            print "Invalid mic specified: ", mic
            sys.exit(errno.ENFILE)
    else:
        if os.path.isfile(args[0]):
            file = os.path.basename(args[0])
            files.append(args[0])
            mic = file.split("_")[0]
        else:
            print "Invalid file specified: ", file
            sys.exit(errno.ENFILE)

    s3cxn = boto.connect_s3()
    bucket_name = BUCKET_PREFIX+mic.lower()
    bucket = check_s3_bucket_exists(s3cxn, bucket_name)    
    if bucket is None:
        print "No such bucket: ", bucket.name 
        if options.create_buckets == True:
            print "Creating s3 bucket: ", bucket_name
            s3cxn.create_bucket(bucket_name)
        else:
            sys.exit(errno.ENFILE)
    
    for f in files:
        print "Uploading: ", f
        s3_multipart_upload.main(f, bucket_name)         
     
        


