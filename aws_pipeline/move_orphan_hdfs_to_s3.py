#!/bin/env python

import boto
import commands
import time
import sys
import run_instances


SSH_COMMAND= "ssh -i ~/aws/capk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "


ec2cxn = boto.connect_ec2()
s3cxn = boto.connect_s3()

reservations = ec2cxn.get_all_instances()

#print reservations

running_instances = []

for r in reservations:
    #print "Checking reservation: ", r
    for inst in r.instances:
        inst.update() 
        #print "Checking instance: ", inst.id, inst.state
        if inst.state == "running":
            #print inst.id, " - running"
            running_instances.append(inst)
   
instance_files = dict()
hdf_files = []
for i in running_instances:
    print "Checking for files on: ", i.dns_name
    command = SSH_COMMAND + "ec2-user@"+i.dns_name+" \" ls /home/ec2-user/features/ \" "
    (retcode, output) = commands.getstatusoutput(command)
    if retcode != 0:
        print "Found no files on: ", i.dns_name
        continue
    #print output
    hdf_files = []
    files =  output.split('\n')
    for f in files:
        if f.split('.')[-1] == "hdf":
            hdf_files.append(f)
            bucketname = run_instances.s3_bucketname_from_filename(f)    
            
            #if check_s3_files_exist(s3cxn, bucketname, [f])
            #    continue;
            if not run_instances.check_s3_files_exist(s3cxn, bucketname, [f]):
                run_instances.hdf_to_s3(i, f, bucketname)
            else:
                print "Not moving: ", f
    instance_files[i] = hdf_files

print instance_files

sys.exit(0)

