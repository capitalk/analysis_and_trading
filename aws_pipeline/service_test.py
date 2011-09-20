#!/usr/bin/env python

import sys
import boto
import boto.utils

if __name__ == "__main__":
    md = boto.utils.get_instance_metadata()
    msg = "OK: %s" % md['instance-id']
    sys.write.stdout(msg) 
