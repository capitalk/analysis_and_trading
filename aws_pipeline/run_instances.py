#!/usr/bin/env python
"""Start feature extraction jobs on AWS for processing into hdf files.

Usage:
run_instances.py <filename or directory> -a AWS image name -n number of instances [-o output queue] [-i input queue] 

"""
import os
import sys
import glob
import subprocess
import copy
import contextlib
import collections
import functools
import commands
import time
import multiprocessing
from subprocess import Popen
from multiprocessing.pool import IMapIterator
from optparse import OptionParser
import s3_download_file
import s3_multipart_upload
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
                        FileTransferSpeed, FormatLabel, Percentage, \
                        ProgressBar, ReverseBar, RotatingMarker, \
                        SimpleProgress, Timer
import boto
import boto.ec2.blockdevicemapping
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.sqs.message import MHMessage


SSH_COMMAND= "ssh -i ~/analysis_and_trading//aws/capk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "

BUCKET_PREFIX="capk-"

USE_EPHEMERAL = False

# First ephemeral storage mount - MUST HAVE TRAILING SLASH
if USE_EPHEMERAL:
    EPHEMERAL0="/media/ephemeral0/"
else:
    EPHEMERAL0="/home/ec2-user/"

# Feature output directory - MUST HAVE TRAILING SLASH
FEATURE_DIR=EPHEMERAL0+"features/"

SECURITY_GROUPS = ['capk']
#INSTANCE_TYPE = "t1.micro"
INSTANCE_TYPE = "c1.xlarge"

def start_instances(ec2cxn, instance_count, image_id):
    print "Attempting to start ", instance_count, " instances of image: ", image_id     
    if USE_EPHEMERAL:
        map = BlockDeviceMapping()
        sdb1 = BlockDeviceType()
        sdb1.ephemeral_name = 'ephemeral0'
        map['/dev/sdb1'] = sdb1
        reservation = ec2cxn.run_instances(image_id, min_count=1, max_count=instance_count, block_device_map=map, security_groups=SECURITY_GROUPS, key_name="capk", instance_type=INSTANCE_TYPE)
    else:
        reservation = ec2cxn.run_instances(image_id, min_count=1, max_count=instance_count, security_groups=SECURITY_GROUPS, key_name="capk", instance_type=INSTANCE_TYPE)

    instances = reservation.instances
    if len(instances) == instance_count:
        print "Started ", instance_count, " instances"
    
    return instances

    

def main(args, inqueue, outqueue, instance_count, image_id):
    s3cxn = boto.connect_s3()
    if s3cxn is None:
        print "s3 connection failed"
        raise Error("s3 connection failed")
    ec2cxn = boto.connect_ec2()

    if ec2cxn is None:
        print "ec2 connection failed"
        raise Error("ec2 connection failed")

    sqscxn = boto.connect_sqs()
    if sqscxn is None:
        print "sqs connection failed"
        raise Error("sqs connection failed")

    instances = start_instances(ec2cxn, instance_count, image_id)
    for i in instances:
        print "==>", i.id

    print "Waiting for all instances to enter running state"

    while True :
        non_running = filter(filter_non_running, instances)
        #print non_running
        if len(non_running) > 0:
            for instance in instances:
                try:
                    instance.update()
                    if instance.state == "terminated":
                        print "Instance " + instance + " unexpectedly entered terminated state - exiting"
                        sys.exit(3)
                except:
                    #print "Exception updating instance - exiting"
                    #sys.exit(4)
                    continue
                #print "Checking state: ", instance.id, instance.state
            print "Waiting 5 seconds"
            time.sleep(5)
        else:
            break
            
    print "All instances seem to be running - waiting  for services to start"
    # Do a stupid check to see if we can ssh in

    s = instances[:]
            
    #for i in xrange(len(s) - 1, -1, -1):
    while len(s) > 0:
        x = len(s) - 1
        print "Checking services on instance: ", s[x]
        command = SSH_COMMAND + "ec2-user@"+s[x].dns_name+ " \" ls  \""
        process = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = process.communicate()
        if len(out) > 0:
            del s[x] 
        else:
            print "Waiting 10 seconds"
            time.sleep(10)
        
    print "Services started"
    print "Starting jobs" 


    for inst in instances:
        command = SSH_COMMAND + " ec2-user@"+inst.dns_name+ " \"source /home/ec2-user/.bash_profile && python /home/ec2-user/analysis_and_trading/aws_pipeline/process_ticks.py -i %s -o %s \""
        Command = command % (inqueue, outqueue)
        print "Command: ", Command
        #(code, output)  = commands.getstatusoutput(Command)
        sp = Popen(Command, shell=True)
        print "Popen result: ", sp
                
    outq = sqscxn.create_queue(outqueue)
    m = MHMessage()
    outq.set_message_class(MHMessage)
    retrys = 0
    retry_wait = 60
    while retrys < 10:
        rs = outq.get_messages() 
        if len(rs) >= 1:
            m = MHMessage()
            m = rs[0]
            print "Received message: ", m.get_body()
            outq.delete_message(m)
            retrys = 0 
        else:
            time.sleep(retry_wait)
            retrys += 1


def filter_non_running(i):
    """Used to filter the list of non-running instances - anything "pending" or terminated" will be removed"""
    return i.state != "running"    

def extract_features(instance, file):
    command = SSH_COMMAND + " ec2-user@"+instance.dns_name+" \"source /home/ec2-user/.bash_profile     && python /home/ec2-user/analysis_and_trading/feature_extraction/extractFeatures.py -d "+FEATURE_DIR+"  /home/ec2-user/"+file+" \" "
    sp = Popen(command, shell=True)
    return sp


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--outqueue", dest="outqueue", help="SQS outbound queue name", default='outq')
    parser.add_option("-i", "--inqueue", dest="inqueue", help="SQS inbound queue name", default='inq')
    parser.add_option("-n", "--instances", dest="instance_count", help="Number of EC2 instances", default='2')
    parser.add_option("-a", "--image", dest="image_id", help="Image id", default=None)
 
    (options, args) = parser.parse_args()

    if options.image_id is None:
        print "ERROR - Must contain an image identifier to launch"
        print __doc__
        sys.exit()
        
    kwargs = dict(inqueue = options.inqueue, outqueue = options.outqueue, instance_count = options.instance_count, image_id = options.image_id)
    main(args, **kwargs)


