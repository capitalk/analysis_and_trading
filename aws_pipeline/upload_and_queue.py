#!/usr/bin/env python
"""Move a set of ticks in csv.gz format to s3. Can take directory or file

Usage:
s3_move_ticks.py <directory_or_file> <-q queue_name> [-f create buckets if they don't exist] 

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
from boto.sqs.message import MHMessage


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
    parser.add_option("-f", "--create-buckets", action='store_true', dest="create_buckets", help="Create buckets on s3 if they don't exist now", default=False)
    parser.add_option("-q", "--queue", dest="queue", help="SQS queue name", default="inq")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        print __doc__
        sys.exit()
        exit(-1)

    if options.queue is None:
        print __doc__
        exit(-1)

    kwargs = dict(mode="test")
    print args
    #path = args[0]
    files = []
    for arg in args:
        if not os.path.exists(arg):
            print "Specified path does not exist: ", arg 
            continue 
        if os.path.isdir(arg):
            if arg[-1] != os.path.sep:
                arg=arg+os.path.sep
            #files.append(glob.glob(arg+"*.csv.gz"))
            files = files + glob.glob(arg+"*.csv.gz")
            #mic = path.split(os.path.sep)[-3]
            #if mic not in MIC_LIST:
                #print "Invalid mic specified: ", mic
                #sys.exit(errno.ENFILE)
        else:
            if os.path.isfile(args[0]):
                #files.append(glob.glob(arg))
                files = files + glob.glob(arg)
                #file = os.path.basename(args[0])
                #mic = file.split("_")[0]
            else:
                print "Invalid file specified: ", file
                #sys.exit(errno.ENFILE)

    print "Processing files: ", files 

    s3cxn = boto.connect_s3()
    sqscxn = boto.connect_sqs()
    exists = False
    q = sqscxn.create_queue(options.queue)
    q.set_message_class(MHMessage)
    
    # Check that all buckets exist or create them if needed
    for f in files:
        basefile = os.path.basename(f)
        mic = basefile.split("_")[0]
        bucket_name = BUCKET_PREFIX+mic.lower()
        try:
            print "Checking bucket: ", bucket_name
            bucket = check_s3_bucket_exists(s3cxn, bucket_name)    
        except Exception:
            if options.create_buckets == True:
                print "Creating bucket: ", bucket_name
                s3cxn.create_bucket(bucket_name)
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

        m = MHMessage()
        m['input_file'] = os.path.basename(f)
        m['bucket'] = bucket_name
        print "Queueing message" , m.get_body(), " ==> ", options.queue
        q.write(m)
        
        
     
        

