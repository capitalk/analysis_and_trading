from optparse import OptionParser
from subprocess import Popen, PIPE
import time
import sys

SSH_COMMAND= "ssh -i ~/aws/capk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "

#running_procs = [ Popen(["/usr/bin/ssh -i /home/timir/aws/capk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ec2-user@50.16.228.182 \" python /home/ec2-user/finishFile.py \""], shell=True)]
#running_procs = [ Popen([SSH_COMMAND+ " ec2-user@50.16.228.182 python /home/ec2-user/finishFile.py "], stdout=PIPE, stderr=PIPE)]
#running_procs = [ Popen(["/bin/ls"])]

parser = OptionParser()
(options, args) = parser.parse_args()
if len(args) < 2:
    sys.exit(0)

ip1 = args[0]
ip2 = args[1] 

sp1 = Popen(SSH_COMMAND + "ec2-user@"+ip1 + " python /home/ec2-user/finishFile.py & ", shell=True);
sp2 = Popen(SSH_COMMAND + "ec2-user@"+ip2 + " python /home/ec2-user/finishFile.py & ", shell=True);

f1 = "TEST1.csv.gz"
f2 = "TEST2.csv.gz"

d = dict()

d[sp1] = f1
d[sp2] = f2

while d:
    for sp in d:
        retval = sp.poll()
        print retval
        if retval is None:
            time.sleep(1)
        else: 
            del(d[sp])
            print "Remove", sp
            break

sys.exit(0)

while running_procs:
    for proc in running_procs:
        #print proc
        retcode = proc.poll()
        if retcode is not None: #Process finished
            running_procs.remove(proc)
            break
        else:
            time.sleep(0.1)
            continue
    
if retcode != 0:
    """Error handling"""
    print "Error - retcode: ", retcode
