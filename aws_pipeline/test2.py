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


SSH_COMMAND= "ssh -i ~/aws/capk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "

BUCKET_PREFIX="capk-"

# First ephemeral storage mount - MUST HAVE TRAILING SLASH
#EPHEMERAL0="/media/ephemeral0/"
EPHEMERAL0="/home/ec2-user/"

# Feature output directory - MUST HAVE TRAILING SLASH
FEATURE_DIR=EPHEMERAL0+"features/"


s3cxn = boto.connect_s3()
if s3cxn is None:
    print "s3 connection failed"
    raise Error("s3 connection failed")
ec2cxn = boto.connect_ec2()
if ec2cxn is None:
    print "ec2 connection failed"
    raise Error("ec2 connection failed")

instanceCount = 1

images = ec2cxn.get_all_images(owners="self")
if len(images) == 1:
    image = images[0]
    print "Using image: ", image.name, image.id
    secGroups = ["capk"]
    print "Starting ", instanceCount, " instances"

    map = BlockDeviceMapping()
    sdb1 = BlockDeviceType()
    #sdc1 = BlockDeviceType()
    #sdd1 = BlockDeviceType()
    #sde1 = BlockDeviceType()
    sdb1.ephmeral_name = 'ephemeral0'
    #sdc1.ephmeral_name = 'ephemeral1'
    #sdd1.ephmeral_name = 'ephemeral2'
    #sde1.ephmeral_name = 'ephemeral3'
    map['/dev/sdb1'] = sdb1
    #bdm['/dev/sdc1'] = sdc1
    #bdm['/dev/sdd1'] = sdd1
    #bdm['/dev/sde1'] = sde1

    reservation = ec2cxn.run_instances(image.id, min_count=1, max_count=instanceCount, security_groups=secGroups, key_name="capk", instance_type="c1.xlarge", instance_initiated_shutdown_behavior="stop", block_device_map=map)
    instances = reservation.instances
    print instances

else:
    print "More than one image exists - exiting"
    sys.exit(2)


