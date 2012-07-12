#!/usr/bin/env python
"""Read messages from queue processing files as directed by msg. Files must be on S3 already. 

Usage:

"""
import os
import commands
from optparse import OptionParser
import s3_multipart_upload
import s3_download_file
import boto
import time
import h5py
from boto.sqs.message import MHMessage

feature_extractor = "~/analysis_and_trading/feature_extraction/extractFeatures.py"


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
    parser.add_option("-t", "--terminate", dest="terminate", help="Terminate instance when done", default=False)
    parser.add_option("-e", "--ephemeral", dest="ephemeral", help="Use ephemeral storage on EC2", default=False)
    parser.add_option("-r", "--max_retries", dest="max_retries", help="max number of retries before we accept queue is empty", default=3)
    parser.add_option("-w", "--retry_wait", dest="retry_wait", help="time to sleep between retries", default=10)

    (options, args) = parser.parse_args()

    # First ephemeral storage mount - MUST HAVE TRAILING SLASH
    if options.ephemeral:
      STORAGE_PREFIX = "/media/ephemeral0/"
    else:
      STORAGE_PREFIX = "/home/ec2-user/"

    FEATURE_DIR = STORAGE_PREFIX + "features/"


    # Feature output directory - MUST HAVE TRAILING SLASH
    print "Reading from queue: ", options.inqueue
    print "Writing to queue: ", options.outqueue

    s3cxn = boto.connect_s3()
    sqscxn = boto.connect_sqs()
    qin = sqscxn.create_queue(options.inqueue)
    qout = sqscxn.create_queue(options.outqueue)
    qin.set_message_class(MHMessage)
    qout.set_message_class(MHMessage)
    retries = 0
    while retries < options.max_retries:
        result_set = qin.get_messages(1, visibility_timeout=60*30)
        if len(result_set) >= 1:
            m = result_set[0] 
            print "Received message: ", m.get_body()
            input_file = m['input_file'] 
            bucket = m['bucket'] 
            s3_download_file.get_s3_file_to_local(s3cxn, bucket, input_file, input_file, STORAGE_PREFIX)	

            full_input_path = STORAGE_PREFIX + input_file
            hdf_file = input_file.replace('csv.gz', 'hdf')
            full_hdf_path = FEATURE_DIR + hdf_file
            if os.path.isfile(full_hdf_path) and hdf_complete(full_hdf_path):
                print "HDF generated and complete - skipping feature extraction" 
            else:
                print "Processing file: ", input_file
                command = "python %s -d %s %s" % \
                  (feature_extractor, FEATURE_DIR, full_input_path) 
                (code, string) = commands.getstatusoutput(command) 
                if options.debug is True:
                    print "Processing retured: ", code, string

            print "Moving processed file file to bucket"
            # KTK - TODO - should wrap in try except block - to catch failed upload
            s3_multipart_upload.main(full_hdf_path, bucket)
           
            retries = 0
            md = boto.utils.get_instance_metadata()
            m['instance-id'] = md['instance-id']
            m['public-hostname'] = md['public-hostname']
            m['completion-time'] =  time.asctime(time.gmtime())
            qout.write(m) 
            qin.delete_message(m)
            os.remove(full_hdf_path)
            os.remove(full_input_path)
                 
        else:
            time.sleep(options.retry_wait)
            retries += 1
     
        if retries == options.max_retries:
            md = boto.utils.get_instance_metadata()
            ec2cxn = boto.connect_ec2()
            m = MHMessage()
            m['shutdown-time'] = time.asctime(time.gmtime()) 
            m['instance-id'] = md['instance-id']
            m['public-hostname'] = md['public-hostname']
            qout.write(m)
            if options.terminate:
                ec2cxn.terminate_instances([md['instance-id']]) 



