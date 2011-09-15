#!/usr/bin/python
from optparse import OptionParser
from operator import attrgetter, itemgetter, methodcaller
from time import strftime
from datetime import datetime
import dateutil.parser
import sys
import csv
from orderBook import * 
import orderBookConstants


csvDelimiter = ','
fixLogDelimiter = '\001' 
timeDelimiter = ' '

def read_rcv_time(filename, debug=False):
    csvReader = csv.reader(open(filename, 'r'), delimiter=timeDelimiter)
    nLine = 0
    t1 = datetime.today()
    
    for row in csvReader:
      nLine += 1
      time = row[0]
      #print "Line #", nLine , time
      t0 = dateutil.parser.parse(time)
      #print "t0: ", t0
      #print "t1: ", t1
      tdelta = t1-t0
      if (t1 > t0 and nLine > 2):
        print "Timestamp t1 > t0 by: ", tdelta, 
        print "line: ", nLine, "prev: ", t1, "line: ", nLine-1, "current: ", t0
      else:
        pass
        #print "+" ,
      t1 = t0;
     
def read_snd_time(filename, debug=False):
    csvReader = csv.reader(open(filename, 'r'), delimiter=fixLogDelimiter)
    nLine = 0
    t1 = datetime.today()
    
    for row in csvReader:
      nLine += 1
      time = row[5]
      fix_time_field = "52="
      if (time.startswith(fix_time_field)):
        sndTime = time[len(fix_time_field):]
      #print "Line #", nLine , time, sndTime
      t0 = dateutil.parser.parse(sndTime)
      #print "t0: ", t0
      #print "t1: ", t1
      tdelta = t1-t0
      if (t1 > t0 and nLine > 2):
        print "Timestamp t1 > t0 by: ", tdelta, 
        print "line: ", nLine, "prev: ", t1, "line: ", nLine-1, "current: ", t0, "\n"
      else:
        pass
        #print "+" ,
      t1 = t0;

parser = OptionParser();
parser.add_option("-r" , "--read", dest="fixlogfile", help="Read logs from file", metavar="FILE")
parser.add_option("-d", "--debug", dest="debug",  action="store_true", help="Print debug output")
parser.add_option("-t", "--use_time", dest="use_time",  help="Which timestamp to use - send (snd) or received (rcv) - rcv is our timestamp and send it exchange")

if __name__ == '__main__': 
    (options, args) = parser.parse_args()
    if len(sys.argv) == 1:
      print sys.argv[0], "--help for usage\n" 
      exit(-1)
    if options.debug:
      print "Input: ", options.fixlogfile
    if options.fixlogfile:
        if options.use_time == "rcv":
            read_rcv_time(options.fixlogfile, options.debug)
        if options.use_time == "snd":
            read_snd_time(options.fixlogfile, options.debug)
      #for x in read_time(options.fixlogfile, options.debug):
      #  pass


