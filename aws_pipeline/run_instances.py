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
import atexit 

SSH_COMMAND= "ssh -i ~/.ec2/capk.key -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "

def exec_remote(inst, command, verbose=False):
  local_command = SSH_COMMAND + " ec2-user@" + inst.dns_name + " \"" + command + "\""
  print "Running: ", local_command
  process = Popen(local_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  return process.communicate()


BUCKET_PREFIX="capk-"

SECURITY_GROUPS = ['capk']


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
    # never leave instances running at the end of the script
    def kill_instances():
      for instance in instances:
        instance.update()
        if instance.state != 'terminated':
          print "Killing ", instance 
          instance.terminate()
    atexit.register(kill_instances)

    print "Started ", instance_count, " instances"
    for i in instances:
        print "  =>", i.id
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

    ec2cxn = boto.connect_ec2()
    if ec2cxn is None:
        print "ec2 connection failed"
        raise RuntimeError("ec2 connection failed")
    if options.image_id is None:
        print "ERROR - Must contain an image identifier to launch"
        print "Choose on of the following:" 
        for image in ec2cxn.get_all_images(owners=['self']):
          print "   ", image.id, ":", image.name, "-",  image.architecture
        print
        parser.print_help()
        sys.exit()

    s3cxn = boto.connect_s3()
    if s3cxn is None:
        print "s3 connection failed"
        raise RuntimeError("s3 connection failed")

    sqscxn = boto.connect_sqs()
    if sqscxn is None:
        print "sqs connection failed"
        raise RuntimeError("sqs connection failed")

    instances = start_instances(ec2cxn, options.instance_count, options.image_id,
      use_ephemeral = options.ephemeral, 
      instance_type = options.instance_type)
 
    print "Waiting for all instances to enter running state"

    def instance_is_running(instance):
      return instance.state == "running" 

    def remove_running(instances):
      return filter(lambda i: not instance_is_running(i), instances)

    max_retries = 10
    retry = 0
    wait_time = 5 
    non_running = remove_running(instances)
    while len(non_running) > 0:
        for instance in instances:
          try:
            instance.update()
            if instance.state == "terminated":
               print "Instance " + instance + " unexpectedly entered terminated state - exiting"
               # it's ok to just die here since kill_instances is registered to run atexit
               sys.exit(3)
          except:
            print "Unexpected exception while updating instance - exiting" 
            print sys.exc_info()
            sys.exit(3)
        sys.stdout.write("-")
        sys.stdout.flush()
        time.sleep(wait_time)
        non_running = remove_running(non_running)
        retry += 1
        if retry > max_retries: 
          print "After %d retries there are still %d instances which aren't running" % (max_retries, len(non_running))
          sys.exit(3)
    
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

    

    # try running remote command, return true if it needs to be retried
    def try_remote_exec(instance):
        # first check if we can ls remotely-- if not, then abandon ship 
      try:
        (ls_output, _) = exec_remote(instance, "ls", verbose=False)
        if len(ls_output) == 0:
          return True          
        exec_remote(instance, remote_command, verbose=True)
        return False
      except: 
        return True

      
    n_instances = len(instances)
    unfinished = filter(try_remote_exec, instances)
    retry = 0 
    wait_time = 20
    max_retries = 30 
    while retry < max_retries:
      unfinished = filter(try_remote_exec, unfinished)
      if len(unfinished) == 0:
        break
      else:
        time.sleep(wait_time)
        retry += 1
        sys.stdout.write("-")
        sys.stdout.flush()

    if retry >= max_retries:
      print "Couldn't connect to %d of %d instances, giving up" % ( len(unfinished), n_instances)
      sys.exit(3) 
    
    retry = 0
    retry_wait = 60
    max_retries = 10
    n_done = 0
    while n_done < n_work_items and retry < max_retries:
        m = outq.read() 
        if m is not None:
            print "Received message: ", m.get_body()
            if 'input_file' in m: 
              n_done += 1
            outq.delete_message(m)
            retry = 0
        else:
            time.sleep(retry_wait)
            retry += 1

    if retry >= max_retries: 
      print "Too many retries!" 
      sys.exit(3)
    else:
      print "Done!" 

