#!/usr/bin/env python
"""Start feature extraction jobs on AWS for processing into hdf files.

Usage:
run_instances.py <filename or directory> -a AWS image name -n number of instances [-o output queue] [-i input queue] 

"""
import sys
import subprocess
import time
from subprocess import Popen
from optparse import OptionParser

import boto
import boto.ec2.blockdevicemapping
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.sqs.message import MHMessage


SSH_COMMAND= "ssh -i ~/analysis_and_trading//aws/capk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "

BUCKET_PREFIX="capk-"

SECURITY_GROUPS = ['capk']

def clean_up_and_die(instances, status=0):
  for instance in instances:
    instance.terminate()
  print "Exiting with status", status
  sys.exit(status)

def start_instances(ec2cxn, instance_count, image_id, use_ephemeral = False, instance_type="c1.xlarge"):
    print "Attempting to start ", instance_count, " instances of image: ", image_id     
    if use_ephemeral:
        dev_map = BlockDeviceMapping()
        sdb1 = BlockDeviceType()
        sdb1.ephemeral_name = 'ephemeral0'
        dev_map['/dev/sdb1'] = sdb1
        reservation = ec2cxn.run_instances(
          image_id, min_count=1, 
          max_count=instance_count, 
          block_device_map=dev_map, 
          security_groups=SECURITY_GROUPS, 
          key_name="capk", 
          instance_type=instance_type)
    else:
        reservation = ec2cxn.run_instances(
          image_id, 
           min_count=1, 
           max_count=instance_count, 
           security_groups=SECURITY_GROUPS, 
           key_name="capk", 
           instance_type=instance_type)

    instances = reservation.instances
    print "Started ", instance_count, " instances"
    for i in instances:
        print "==>", i.id
    # TODO: Add a 'kill all these instances' function
    if len(instances) != instance_count:
      print "Expected %d instances, got %d" % (instance_count, len(instances))
    return instances






if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--outqueue", dest="outqueue", help="SQS outbound queue name", default='outq')
    parser.add_option("-i", "--inqueue", dest="inqueue", help="SQS inbound queue name", default='inq')
    parser.add_option("-n", "--instances", dest="instance_count", help="Number of EC2 instances", default=2)
    parser.add_option("-a", "--image", dest="image_id", help="Image id", default=None)
    parser.add_option("-e", "--ephemeral", dest="ephemeral", help="Use ephemeral storage on EC2", default=False)
    parser.add_option("-t", "--instance_type", dest="instance_type", help="How big of an instance do you want?", default="c1.xlarge")

    (options, args) = parser.parse_args()

    if options.image_id is None:
        print "ERROR - Must contain an image identifier to launch"
        print __doc__
        # TODO: Print a list of our available AMI's 
        sys.exit()

    s3cxn = boto.connect_s3()
    if s3cxn is None:
        print "s3 connection failed"
        raise RuntimeError("s3 connection failed")
    ec2cxn = boto.connect_ec2()

    if ec2cxn is None:
        print "ec2 connection failed"
        raise RuntimeError("ec2 connection failed")

    sqscxn = boto.connect_sqs()
    if sqscxn is None:
        print "sqs connection failed"
        raise RuntimeError("sqs connection failed")

    instances = start_instances(ec2cxn, options.instance_count, options.image_id,
      use_ephemeral = options.ephemeral, 
      instance_type = options.instance_type)

    print "Waiting for all instances to enter running state"

    def instance_is_running(instance):
      if instance.state != "running": 
        return False
      # hackish test to check whether services have started by trying 
      # to SSH and ls 
      else:
        command = SSH_COMMAND + "ec2-user@" + instance.dns_name + " \" ls  \""
        process = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = process.communicate()
        return len(out) > 0

    def remove_running(instances):
      return filter(lambda i: not instance_is_running(i), instances)

    max_retries = 10
    retry = 0
    wait_time = 3 
    non_running = remove_running(instances)
    while len(non_running) > 0:
        for instance in instances:
          try:
            instance.update()
            if instance.state == "terminated":
               print "Instance " + instance + " unexpectedly entered terminated state - exiting"
               clean_up_and_die(instances, 3)
          except:
            print "Unexpected exception while updating instance - exiting" 
            clean_up_and_die(instances, 3)
        print "Waiting %d seconds" % wait_time
        time.sleep(wait_time)
        non_running = remove_running(non_running)
        retry += 1
        if retry > max_retries: 
          print "After %d retries there are still %d instances which aren't running" % (max_retries, len(non_running))
          clean_up_and_die(instances)
    
    print "Services started"

    inq = sqscxn.create_queue(options.inqueue) 
    outq = sqscxn.create_queue(options.outqueue)
    outq.set_message_class(MHMessage)

    n_work_items = inq.count() 
    
    print "Starting jobs" 
    remote_command = "source /home/ec2-user/.bash_profile && python" \
      + "/home/ec2-user/analysis_and_trading/aws_pipeline/process_ticks.py -i %s -o %s --terminate" \
      % (options.inqueue, options.outqueue)
    if options.ephemeral:
      remote_command += " --ephemeral" 

    for inst in instances:
        local_command = SSH_COMMAND + " ec2-user@" + inst.dns_name + " \"" + remote_command + "\""
        print "Running: ", local_command
        sp = Popen(local_command, shell=True)
        print "Popen result: ", sp
                
    retry = 0
    retry_wait = 60
    max_retries = 10
    n_done = 0
    while n_done < n_work_items and retry < max_retries:
        m = outq.read() 
        if m is not None:
            print "Received message: ", m.get_body()
            if 'input_file' in m: n_done += 1
            outq.delete_message(m)
            retry = 0
        else:
            time.sleep(retry_wait)
            retry += 1

    if retry >= max_retries: 
      print "Too many retries!" 
      clean_up_and_die(instances)
    else:
      print "Done!" 

