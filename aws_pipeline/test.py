#!/bin/env python

import boto
import commands
import time
SSH_COMMAND= "ssh -i ~/aws/capk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "


ec2cxn = boto.connect_ec2()

reservations = ec2cxn.get_all_instances()

#print reservations

instances = []

for r in reservations:
    #print "Checking reservation: ", r
    for inst in r.instances:
        inst.update() 
        #print "Checking instance: ", inst.id, inst.state
        if inst.state == "running":
            #print inst.id, " - running"
            instances.append(inst)
   
print instances 

print "All instances seem to be running - waiting  for services to start"
# Do a stupid check to see if we can ssh in


s = instances[:]
"""
s1 = []
for i in s:
        print "Checking service on: ", inst.id
        command = SSH_COMMAND + "ec2-user@"+inst.dns_name+ " \" ls \""
        r = commands.getstatusoutput(command)
        if r[0] == 0:
            print "Removing instance from wait queue: ", inst.id
            s1.remove(inst)
            break
        else:
            time.sleep(5)
        s = s1[:]
print s
"""

for i in xrange(len(s) - 1, -1, -1):
        print "Checking service on: ", s[i].id
        command = SSH_COMMAND + "ec2-user@"+s[i].dns_name+ " \" ls \""
        r = commands.getstatusoutput(command)
        if r[0] == 0:
            print "Removing instance from wait queue: ", s[i].id
            #s.remove(inst)
            del s[i]
            break
        else:
            time.sleep(5)
print s


s = instances[:]
s1 = s[:]
while s:
    for i in s:
        print "Checking service on: ", i.id
        command = SSH_COMMAND + "ec2-user@"+i.dns_name+ " \" ls \""
        r = commands.getstatusoutput(command)
        if r[0] == 0:
            print "Removing instance from wait queue: ", i.id
            s1.remove(i)
            s = s1[:]
            break
        else: 
            time.sleep(5)

print s
        

"""
serviceCheck = instances[:]
while serviceCheck:
    for inst in serviceCheck:
        print "Checking service on: ", inst.id
        command = SSH_COMMAND + "ec2-user@"+inst.dns_name+ " \" ls \""
        r = commands.getstatusoutput(command)
        if r[0] == 0:
            print "Removing instance from wait queue: ", inst.id
            serviceCheck.remove(inst)
            break
        else: 
            time.sleep(5)

def run_remote_command(command)
"""  
