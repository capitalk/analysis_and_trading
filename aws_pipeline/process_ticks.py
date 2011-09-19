#!/usr/bin/env python
"""Read messages from queue processing files as directed by msg. Files must be on S3 already. 

Usage:

"""
import os
import sys
import glob
import subprocess
import errno
import commands
from optparse import OptionParser
import s3_multipart_upload
import s3_download_file
import boto
import mics
from mics import MIC_LIST
from boto.sqs.message import MHMessage

Command = """python /home/ec2-user/analysis_and_trading/feature_extraction/extractFeatures.py -d /home/ec2-user/features/ /home/ec2-user/%s"""
   

# First ephemeral storage mount - MUST HAVE TRAILING SLASH
if USE_EPHEMERAL:
    EPHEMERAL0="/media/ephemeral0/"
else:
    EPHEMERAL0="/home/ec2-user/"

# Feature output directory - MUST HAVE TRAILING SLASH
FEATURE_DIR=EPHEMERAL0+"features/"


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
    parser.add_option("-o", "--outqueue", dest="outqueue", help="SQS outbound queue name", default='outq')
    parser.add_option("-i", "--inqueue", dest="inqueue", help="SQS inbound queue name", default='inq')
    (options, args) = parser.parse_args()

    s3cxn = boto.connect_s3()
    sqscxn = boto.connect_sqs()
    q = sqscxn.create_queue(options.inqueue)
    q = sqscxn.create_queue(options.outqueue)
    q.set_message_class(MHMessage)
    while True:
        rs = q.get_messages(visibility_timeout=60*20)
        if len(rs) > 1:
            input_file = rs['input_file'] 
            bucket = rs['bucket'] 
            get_s3_file_to_local(s3cxn, bucket_name,input_file, input_file, "/home/ec2-user")
            name_parts = input_file.split('.')
            hdf_file = name_parts[0]
            hdf_file += ".hdf"
            command = Command % (input_file)
            commands.getstatusoutput(command) # TODO change to popen
            result = commands.getstatusoutput("source /home/ec2-user/.bash_profile && python /home/ec2-user/analysis_and_trading/aws_pipeline/s3_multipart_upload.py "+FEATURE_DIR+hdf_file+" "+bucket+" \" ")         
             
        else:
            break 
     


