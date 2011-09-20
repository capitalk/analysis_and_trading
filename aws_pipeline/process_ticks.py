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
import time
import h5py
from mics import MIC_LIST
from boto.sqs.message import MHMessage

CommandExtractFiles = """python /home/ec2-user/analysis_and_trading/feature_extraction/extractFeatures.py -d /home/ec2-user/features/ /home/ec2-user/%s"""
CommandFileToS3 = """source /home/ec2-user/.bash_profile && python /home/ec2-user/analysis_and_trading/aws_pipeline/s3_multipart_upload.py  %s %s"""

USE_EPHEMERAL = False

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
    parser = OptionParser()
    parser.add_option("-o", "--outqueue", dest="outqueue", help="SQS outbound queue name", default='outq')
    parser.add_option("-i", "--inqueue", dest="inqueue", help="SQS inbound queue name", default='inq')
    parser.add_option("-d", "--debug", dest="debug", help="Print debug information", default='False')
    (options, args) = parser.parse_args()

    print "Reading from queue: ", options.inqueue
    print "Writing to queue: ", options.outqueue

    s3cxn = boto.connect_s3()
    sqscxn = boto.connect_sqs()
    qin = sqscxn.create_queue(options.inqueue)
    qout = sqscxn.create_queue(options.outqueue)
    qin.set_message_class(MHMessage)
    qout.set_message_class(MHMessage)
    retrys = 0
    retry_wait = 10
    extractOK = False
    moveOK = False
    while retrys < 10:
        rs = qin.get_messages(visibility_timeout=60*30)
        if len(rs) >= 1:
            m = MHMessage()
            m = rs[0] 
            print "Received message: ", m.get_body()
            input_file = m['input_file'] 
            bucket = m['bucket'] 
            s3_download_file.get_s3_file_to_local(s3cxn, bucket,input_file, input_file, EPHEMERAL0)	

            hdf_file = input_file.replace('csv.gz', 'hdf')
            hdf_path = FEATURE_DIR + hdf_file;
            if os.path.isfile(FEATURE_DIR+hdf_file) and hdf_complete(FEATURE_DIR+hdf_file):
                print "HDF generated and complete - skipping feature extraction" 
                extractOK = True
            else:
                print "Processing file: ", input_file
                command = CommandExtractFiles % (input_file)
                (code, string) = commands.getstatusoutput(command) 
                #extractOK = (code == 0)
                if options.debug is True:
                    print "Processing retured: ", code, string

            print "Moving processed file file to bucket"
            # KTK - TODO - should wrap in try except block - to catch failed upload
            s3_multipart_upload.main(FEATURE_DIR+hdf_file, bucket)
            #command = CommandFileToS3 % (FEATURE_DIR+hdf_file, bucket)
            #(code, string) = commands.getstatusoutput(command) 
            #moveOK = (code == 0)
            #if options.debug is True:
            #    print "Move tick files returned: ", code, string
            
            #if extractOK and moveOK:
            retrys = 0
            md = boto.utils.get_instance_metadata()
            m['instance-id'] = md['instance-id']
            m['public-hostname'] = md['public-hostname']
            qout.write(m) 
            qin.delete_message(m)
                 
        else:
            time.sleep(retry_wait)
            retrys += 1
     
        if retrys == 10:
            md = boto.utils.get_instance_metadata()
            ec2cxn = boto.connect_ec2()
            m = MHMessage()
            m['complete-time'] = time.asctime(time.gmtime()) 
            m['instance-id'] = md['instance-id']
            ec2cxn.terminate_instances([md['instance-id']]) 



