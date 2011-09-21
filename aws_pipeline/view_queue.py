#!/usr/bin/env python
"""View the contents of a queue

Usage:
view_queue.py queue_name1 [queue_name2] ... [queue_name3] [-m MessageClass]
NOTE: All queus must use same msgType

"""
import os
import sys
from optparse import OptionParser
import boto
from boto.sqs.message import MHMessage



if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--msgclass", default="boto.sqs.message.MHMessage",  dest="msgclass")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        print __doc__
        sys.exit()

    print "Args: ", args
    
    msgclass = options.msgclass.split('.')[-1]
    temp = options.msgclass.split('.')
    msgpackage = '.'.join(temp[0:-1])
    mod = __import__(msgpackage, fromlist=[msgclass])
    mclass = getattr(mod, msgclass)
    qs = args
    sqscxn = boto.connect_sqs()
    for qname in qs:
        q = sqscxn.create_queue(qname)
        print "====> Stats for queue: ", qname
        nmsg = q.count()
        print "Msg count (est.): ", nmsg
        n = 0
        while n < nmsg:
            q.set_message_class(mclass)
            incoming = mclass()
            rs = q.get_messages(visibility_timeout=60)     
            if len(rs) > 0:
                incoming = rs[0]
                print incoming.get_body()
            n = n+1
    
        
        
     
        


