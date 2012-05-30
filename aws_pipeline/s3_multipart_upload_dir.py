#!/usr/bin/env python
"""Move a set of ticks in csv.gz format to s3. Can take directory or file

Usage:
s3_move_ticks.py <directory_or_file> <bucket_name> [-f create buckets if they don't exist] 

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
    parser.add_option("-f", "--create-buckets", action='store_true', dest="create_buckets", help="Create buckets on s3 if they don't exist now", default=False)
    (options, args) = parser.parse_args()

    if len(args) < 1:
        print __doc__
        sys.exit()
        exit(-1)

    kwargs = dict(mode="test")
    print "Args: ", args
    files = []
    path=args[0];
    bucket_name=args[1];
    if not os.path.exists(path):
        print "Specified path does not exist: ", path 
        exit(-1);
    if os.path.isdir(path):
        if path[-1] != os.path.sep:
            path=path+os.path.sep
        files = files + glob.glob(path+"*")
    else:
        if os.path.isfile(path[0]):
            files = files + glob.glob(path)
        else:
            print "Invalid file specified: ", file

    print "Processing files: ", files 

    s3cxn = boto.connect_s3()
    
    # Check that all buckets exist or create them if needed
    for f in files:
        basefile = os.path.basename(f)
        try:
            print "Checking bucket: ", bucket_name
            bucket = check_s3_bucket_exists(s3cxn, bucket_name)    
        except Exception:
            if options.create_buckets == True:
                print "Creating bucket: ", bucket_name
                bucket = s3cxn.create_bucket(bucket_name)
                bucket = check_s3_bucket_exists(s3cxn, bucket_name)    
            else:
                sys.exit(errno.ENFILE)
    
        bucket = s3cxn.get_bucket(bucket)
        key = bucket.get_key(basefile)
        exists = (key is not None)
        if exists == True:
            print "Key exists - skipping upload"
        else:
            print "Uploading: ", f
            s3_multipart_upload.main(f, bucket_name)         
