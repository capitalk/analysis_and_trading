#!/usr/bin/env python
"""Delete all files matching regex from bucket specified

Usage:
s3_delete_files_from_bucket.py <regex> <bucket> 

Options:
[-i use interactive mode - i.e. confirm all deletes] 
[-v use verbose mode - i.e. display lots of stuff] 
[-f force delete - i.e. just delete and dont ask] 

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


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--confirm-delete", action='store_true', dest="confirm_delete", help="Confrim deletes one by one", default=True)
    parser.add_option("-f", "--force-delete", action='store_false', dest="confirm_delete", help="Confrim deletes one by one")
    parser.add_option("-v", "--verbose", action='store_true', dest="verbose", help="Show verbose output", default=False)
    (options, args) = parser.parse_args()

    confirm_delete = True;
    if options.confirm_delete is False:
        confirm_delete = False;

    if options.confirm_delete is True:
        print "Confirming deletes"



    if len(args) < 1:
        print __doc__
        sys.exit()
        exit(-1)

    print "Args: ", args

    fn_regex = args[0]
    bucket_name = args[1]

    s3cxn = boto.connect_s3()
    exists = False

    bucket = check_s3_bucket_exists(s3cxn, bucket_name)    
    if bucket is False:
            print "No such bucket"
            exit;
            
    bucket = s3cxn.get_bucket(bucket)
    keys = bucket.get_all_keys()
    if len(keys) == 0:
        print "There are no keys in %s bucket" % bucket_name
        exit;
        
    print "Examining %d keys" % len(keys),

    delete_bytes = 0

    names = [fn.name  for fn in keys]
    match_names = fnmatch.filter(names, fn_regex);
    print "of which %d match regex" %  len(match_names)

    for n in match_names:
        if options.confirm_delete is True:
            input = raw_input("PERMANENTLY delete %s from %s bucket? " % (n, bucket_name))
            if input == "y" or input == "Y":
                key = bucket.get_key(n)
                delete_bytes += key.size
                if options.verbose is True: print "Deleting: %s" % n
                bucket.delete_key(n);
            else:
                if options.verbose is True: print "NOT deleting %s" % n
                continue
        else:
            key = bucket.get_key(n)
            delete_bytes += key.size
            if options.verbose: print "Deleting(2): %s" % n
            bucket.delete_key(n);


    print "Deleted %d bytes (%d kB) (%d MB) (%d GB)" % (delete_bytes, delete_bytes/1000, delete_bytes/1000000, delete_bytes/1000000000)
