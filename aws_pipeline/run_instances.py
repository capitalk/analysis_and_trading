#!/usr/bin/env python
"""Start feature extraction jobs on AWS for processing into hdf files.

Usage:
run_instances.py <filename or directory> 

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


SSH_COMMAND= "ssh -i ~/aws/capk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "

BUCKET_PREFIX="capk-"

USE_EPHEMERAL = False

# First ephemeral storage mount - MUST HAVE TRAILING SLASH
if USE_EPHEMERAL:
    EPHEMERAL0="/media/ephemeral0/"
else:
    EPHEMERAL0="/home/ec2-user/"

# Feature output directory - MUST HAVE TRAILING SLASH
FEATURE_DIR=EPHEMERAL0+"features/"

def start_intances(ec2cxn, instance_count):
    images = ec2cxn.get_all_images(owners="self")
    #print "Found ", len(images) , " images"
    if len(images) == 1:
        image = images[0]
        print "Using image: ", image.name, image.id
        secGroups = ["capk"]
        print "Starting ", instance_count, " instances"
        
        if USE_EPHEMERAL:
            map = BlockDeviceMapping()
            sdb1 = BlockDeviceType()
            sdb1.ephemeral_name = 'ephemeral0'
            map['/dev/sdb1'] = sdb1
            reservation = ec2cxn.run_instances(image.id, min_count=1, max_count=instance_count, block_device_map=map, security_groups=secGroups, key_name="capk", instance_type="c1.xlarge")
        else:
            reservation = ec2cxn.run_instances(image.id, min_count=1, max_count=instance_count, security_groups=secGroups, key_name="capk", instance_type="c1.xlarge")

        instances = reservation.instances
        return instances
    else:
        print "More than one image exists - exiting"
        sys.exit(2)
    
    

def main(files, inqueue, outqueue, instance_count):
    s3cxn = boto.connect_s3()
    if s3cxn is None:
        print "s3 connection failed"
        raise Error("s3 connection failed")
    ec2cxn = boto.connect_ec2()
    if ec2cxn is None:
        print "ec2 connection failed"
        raise Error("ec2 connection failed")

    sqscxn = boto.connectsqs()
    if sqscxn is None:
        print "sqs connection failed"
        raise Error("sqs connection failed")

    instances = start_instances(ec2cxn, instance_count)

"""
    basefiles = [os.path.basename(f) for f in files]
    for b in basefiles:
        bucketname = s3_bucketname_from_filename(b)
        print b, bucketname
        if not check_s3_files_exist(s3cxn, bucketname, [b]):
            print "S3 file missing: ", b 
            sys.exit(0)

"""
    print "Waiting for all instances to enter running state"
    while True :
        non_running = filter(filter_non_running, instances)
        print non_running
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
                print "Checking state: ", instance.id, instance.state
            time.sleep(5)
            print "."
        else:
            break
            
    print "All instances seem to be running - waiting  for services to start"
    # Do a stupid check to see if we can ssh in
"""
    if len(instances) == 1:
        while True:
            print "Checking services on SINGLE instance: ", instances
            command = SSH_COMMAND + "ec2-user@"+instances[0].dns_name+ " \" ls  \""
            r = commands.getstatusoutput(command)
            print r[1]
            if r[0] == 0:
                break
            else:
                time.sleep(5)
"""      

    s = instances[:]
    for i in xrange(len(s) - 1, -1, -1):
        print "Checking services on instance: ", s
        command = SSH_COMMAND + "ec2-user@"+s[i].dns_name+ " \" ls  \""
        r = commands.getstatusoutput(command)
        print r[1]
        if r[0] == 0:
            print "Removing instance from wait queue: ", s[i].id
            del s[i]
            break
        else:
            time.sleep(5)
        
    print "Services started"
    print "Starting jobs" 


    foreach inst in instances:
        command = SSH_COMMAND + " ec2-user@"+inst.dns_name+ " \"source /home/ec2-user/.bash_profile && python /home/ec2-user/analysis_and_trading/aws_pipeline/process_ticks.py -i %s -o %s "
        Command = command % (inquque, outqueue)
        result = commands.getstatusoutput(Command)
                
    outq = sqscxn.create_queue(outqueue)
    m = MHMessage()
    outq.set_message_class(MHMessage)
    retrys = 0
    retry_wait = 10
    while retrys < 10:
        rs = outq.get_messages() 
        if len(rs) >= 1:
            m = MHMessage()
            m = rs[0]
            print "Received message: ", m.get_body()
            outq.delete_message(m)
    
        else:
            time.sleep(retry_wait)
            retrys += 1

"""
    jobq = collections.deque()
    instance_file = zip(instances, basefiles)
    instance_file_dict = dict(instance_file)

    for inst in instance_file_dict:
        file = instance_file_dict[inst]
        bucketname = s3_bucketname_from_filename(file)
        if prefetch_file(inst, bucketname, file) == False:
            print "Error fetching file - terminating instance";
            terminate_list = [inst.id]
            ec2cxn.terminate_instances(terminate_list);
        print "Processing ", file, " on : ", inst.dns_name, "\n"
        sp = extract_features(inst, file)
        job = [sp, inst, file]
        jobq.append(job)
        print "Adding ", sp, " to ", inst

    
    while jobq:
        [proc, inst, file] = jobq.pop();
        print "Polling", proc, inst, file 
        retcode = proc.poll()
        if retcode is not None:
            print "Feature extraction complete on: ", file, inst.dns_name
            bucketname = s3_bucketname_from_filename(file)
            print "Moving ", file, " to bucket ", bucketname
            hdf_to_s3(inst, file, bucketname)
            break
        else:
            jobq.appendleft([proc, inst, file])
            print "Sleeping"
            time.sleep(60*10)
            continue
"""


def s3_bucketname_from_filename(filename):
    names = filename.split(".")
    basename = names[0]
    [mic, symbol, year, month, day] = basename.split('_')
    bucketname = BUCKET_PREFIX+mic.lower()
    return bucketname

def s3_key_from_filename(s3cxn, filename):
    bucketname = s3_bucketname_from_filename(filename)
    bucket = s3cxn.get_bucket(bucketname)
    if bucket is None:
        print "No such bucket: ", bucketname
        return None
    else:
        keyname = s3_filename_from_attrs(mic, symbol, year, month, day)
        key = bucket.get_key(keyname)
    return key     
    

def s3_filename_from_attrs(mic, symbol, year, month, day):
    bucketname = mic.lower()
    filename = mic.upper()+pair.upper()+"_"+year+"_"+month+"_"+day
    return filename
        
 
def check_s3_key_exists(s3cxn, bucket_name, key):
    if bucket_name is None:
        print "No bucket specified"
        return False
    bucket = s3cxn.get_bucket(bucket_name)
    if bucket is None:
        print "No bucket found on S3 with name: ", bucket_name
        return False
    k = bucket.get_key(key)
    if k is None:
        print "No key found on S3: ", key
        return False
    else:
        return True

def check_s3_files_exist(s3cxn, bucket, files):
    for f in files:
        print "Checking:", bucket, " for ", f
        basename = os.path.basename(f)
        bucket = s3cxn.get_bucket(bucket)
        if bucket is None:
            print "No such bucket: ", bucket
            return False
        key = bucket.get_key(basename)
        ret = check_s3_key_exists(s3cxn, bucket, basename)
        if ret is None:
            print "No such key (file): ", basename
            return False
        return ret 

def hdf_to_s3(instance, file, bucket):
    """Move a file to a given bucket - set the key name to be the same as the file"""
    name_parts = file.split('.')
    hdf_file = name_parts[0]
    hdf_file += ".hdf"
    result = commands.getstatusoutput(SSH_COMMAND + " ec2-user@"+instance.dns_name+ " \"source /home/ec2-user/.bash_profile && python /home/ec2-user/analysis_and_trading/aws_pipeline/s3_multipart_upload.py "+FEATURE_DIR+hdf_file+" "+bucket+" \" ")
    print "Move hdf result: ", result
    return result[0] == 0
    
            
def prefetch_file(instance, bucketname, keyname):
    """Prefetch a csv file to a given instance"""
    file = os.path.basename(keyname)
    print "Fetching: ", keyname, " on ", instance.dns_name
    command = SSH_COMMAND + " ec2-user@"+instance.dns_name+ " \"source /home/ec2-user/.bash_profile &&  python /home/ec2-user/analysis_and_trading/aws_pipeline/s3_download_file.py -d "+EPHEMERAL0+" -b "+bucketname+" -k " + file + "\"" 
    print command
    result = commands.getstatusoutput(command)
    #print "Fetch result:", result
    if result[0] != 0:
        print result[1]
    
    return result[0] == 0

def check_job_complete(instance, file):
    """Run script to check for finished flag in hdf file""" 
    # TODO - use SQS/SNS for job completion
    name_parts = file.split('.')
    hdf_file = name_parts[0]
    hdf_file += ".hdf"
    command = SSH_COMMAND + " ec2-user@"+instance.dns_name+ " \"source /home/ec2-user/.bash_profile && python /home/ec2-user/analysis_and_trading/aws_pipeline/testHdfFinished.py "+FEATURE_DIR+hdf_file+" \" "
    print command
    result = commands.getstatusoutput(command)
    print "Check complete result: ", result
    if (result[0] == 0):
        print "Job finished on: ", instance.dns_name
    return result[0] == 0 


def filter_non_running(i):
    """Used to filter the list of non-running instances - anything "pending" or terminated" will be removed"""
    return i.state != "running"    

def extract_features(instance, file):
    command = SSH_COMMAND + " ec2-user@"+instance.dns_name+" \"source /home/ec2-user/.bash_profile     && python /home/ec2-user/analysis_and_trading/feature_extraction/extractFeatures.py -d "+FEATURE_DIR+"  /home/ec2-user/"+file+" \" "
    sp = Popen(command, shell=True)
    return sp

def get_ec2_instances(ec2cxn, state=None):
    if ec2cxn is None:
        raise RuntimeError("Invalid ec2 connection")
    # how do we distinguish reservations
    reservations = ec2cxn.get_all_instances()
    instances = []
    for r in reservations:
        for inst in r.instances:
           if state is None or  inst.state == state: 
                instances.append(inst) 
    return instances

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--outqueue", dest="outqueue", help="SQS outbound queue name", default='outq')
    parser.add_option("-i", "--inqueue", dest="inqueue", help="SQS inbound queue name", default='inq')
    parser.add_option("-n", "--instances", dest="instance_count", help="Number of EC2 instances", default='2')
 
    (options, args) = parser.parse_args()
    if len(args) < 1:
        print __doc__
        sys.exit()
    print args
    path = args[0]
    if not os.path.exists(path):
        print "Specified path does not exist: ", path 
        sys.exit(0)
    if os.path.isdir(path):
        if path[-1] != '/':
            path=path+"/"
        files = glob.glob(path+"*.csv.gz")
        args = files
        #args.extend(files)
        
    print "Args: ", args
    print "Options: ", options
    kwargs = dict(outqueue = options.outqueue, inquque = options.inqueue, instance_count = options.instance_count)
    main(args, **kwargs)


