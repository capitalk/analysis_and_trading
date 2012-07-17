#!/usr/bin/python
from optparse import OptionParser
from operator import attrgetter, itemgetter, methodcaller
from time import strftime
from datetime import datetime

import re 
import sys
import csv
from orderBook import Action, Order, OB
from orderBook import ADD_ACTION_TYPE, MODIFY_ACTION_TYPE, DELETE_ACTION_TYPE
from orderBook import BID_SIDE, OFFER_SIDE 
import orderBookConstants
import progressbar
import os
import numpy 
######################
# Version 1
######################
# 20110501-23:59:59.565, EUR/NZD, Q, 1, 2, 1.83429, 1000000, 1
# 0 = datetime
# 1 = currency
# 2 = quote or trade
# 3 = side
# 4 = level 
# 5 = price
# 6 = size
# 7 = num orders in level 
# 8 = [, separated list of size in #7 with order sizes]


######################
#  Version 3

# header
# actions = {A|D|M}, side, price, volume 
# orderbook header = OB, ccy, monotonic timestamp, exchange timestamp
# orderbook entry = {Q|D}, monotonic timestamp, side, level, price, size, time sorted list of order sizes
# Example: 
#
# V3,EUR/USD,0
# A,0,1.36307,500000
# M,0,1.36307,500000
# D,1,1.3631,250000
# OB,EUR/USD,3554600:273218531,1315404477:864000000
# Q,3554600:273218531,0,1,1.36307,1000000,500000,500000
# Q,3554600:270045798,1,1,1.3631,1000000,1000000


obc = orderBookConstants 

#csvDelimiter = ','
fixLogDelimiter = '\001' 
timeDelimiter = ' '

import subprocess
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


#given a file, read groups of lines but trim the groups to not split orderbooks
def line_group_generator(f, sizehint=1000000):
    tail = [] 
    while True:
        lines = f.readlines(sizehint)
        n = len(lines)
        if n == 0: break
        # scan backward til beginning of last orderbook
        idx = n - 1
        while idx > 0:
            currline = lines[idx]
            if currline[0:9] == 'ORDERBOOK': break 
            idx -= 1
        tail.extend(lines[0:idx])
        yield tail 
        tail = lines[idx:] 
        
        
# split up the lines of a data file without splitting up an orderbook
def group_lines(lines, chunksize=100000):
    groups = []
    n = len(lines)
    start = 0 
    stop = min(chunksize, n) - 1
    while start < n: 
        # scan forward until stop is at the end of an orderbook
        while stop < n and lines[stop][0:9] != 'ORDERBOOK': stop +=1 
        group = lines[start:stop-1]
        groups.append(group)
        start = stop
        stop = min(n, start + chunksize) 
    return groups

def read_file_lines(fileobj, max_lines=None):
    if max_lines:
        # assume 200 bytes per line max 
        lst = fileobj.readlines(200*max_lines)
        lst = lst[0:max_lines]
    else:
        lst = fileobj.readlines()
    return lst

def read_lines(filename):
    lines = [] 
    with open(filename, 'r') as f: 
        lines = read_file_lines(f)
    return lines

def parse_datetime(s):
    second = int(s[15:17])
    millisecond = int(s[18:21])
    year = int(s[0:4])
    month = int(s[4:6])
    day = int(s[6:8])
    hour = int(s[9:11])
    minute = int(s[12:14])
    return datetime(year, month, day, hour, minute, second, 1000* millisecond)

# since many datetimes are repeats, 
# cache last datetime and last millisecond string,
lastMillisecond = None
lastSecond = None 
lastDatetime = None 
def parse_datetime_opt(s):
    global lastDatetime 
    global lastMillisecond
    global lastSecond
    second = int(s[15:17])
    millisecond = int(s[18:21])
    
    if lastDatetime and second == lastSecond and millisecond == lastMillisecond:
        return lastDatetime
    else:
        lastSecond = second
        lastMillisecond = millisecond
        year = int(s[0:4])
        month = int(s[4:6])
        day = int(s[6:8])
        hour = int(s[9:11])
        minute = int(s[12:14])
        d = datetime(year, month, day, hour, minute, second, 1000* millisecond)
        lastDatetime = d
        return d

import gc
def books_from_lines_v1(lines, debug=False, end=None, drop_out_of_order=False):
    currBook = None
    # keep track of which side the book starts on,
    # if that side repeats we've reached a new book 
    startSide = None 
    nLine = 0
    nBooks = 0
    
    keep_out_of_order = not drop_out_of_order
    maxTimestamp = None
    book_list = [] 
    for line in lines:
        if end and nBooks > end:           
            break 
        nLine += 1
        if line[0:9] == "ORDERBOOK":
            nBooks += 1
            if currBook is not None: 
                if keep_out_of_order or currBook.lastUpdateTime == maxTimestamp: 
                    book_list.append(currBook)
            timestr = line[10:] 
            lastUpdateTime = parse_datetime_opt(timestr)
            if maxTimestamp is None or lastUpdateTime > maxTimestamp:
                maxTimestamp = lastUpdateTime
            currBook = OB(lastUpdateTime = lastUpdateTime)
        else: 
            row = line.split(',')
            side = row[obc.SIDE]
            entry = Order(
                timestamp = parse_datetime_opt(row[obc.TIMESTAMP]), 
                side = side, 
                level = int(row[obc.LEVEL]), 
                price = float(row[obc.PRICE]), 
                size = long(row[obc.SIZE]), 
                #orderdepthcount = int(row[obc.ORDERDEPTHCOUNT])
                #ccy = row[obc.CURRENCY]
            )
            if (side == obc.BID): currBook.add_bid(entry)
            elif (side == obc.OFFER): currBook.add_offer(entry)
    return book_list 



#comma_or_colon = re.compile(',|:')

class V3_Parser:
    def __init__(self): 
        self.header = {} 
        self.currBook = None
        self.books = [] 
        # keep this around for printing debug info on restarts
        self.actions = []    
        self.SECONDS_PER_DAY = 60 * 60 * 24     
        self.order_cache = {}
    
    def parse_header(self, line): 
        # make sure nothing else has been parsed yet 
        assert self.currBook == None 
        assert len(self.books) == 0 
        assert self.actions == [] 
        assert len(self.header) == 0 
        
        v, ccy, _ = line.split(',')
        assert v == 'V3'
        self.header['ccy'] = ccy
        #header['depth'] = int(maxdepth.strip())
        
    def parse_add_action(self, line):
        _, side, volume, price, _ = line.split(',')
        
        self.actions.append(
            Action(
                action_type = ADD_ACTION_TYPE,
                side = OFFER_SIDE if side == '1' else BID_SIDE,
                price = float(price), 
                volume = int(float(volume))
            )
        )
    
    def parse_delete_action(self, line):
        _, side, volume, price, _ = line.split(',')
        self.actions.append(
            Action(
                action_type = DELETE_ACTION_TYPE,
                side = OFFER_SIDE if side == '1' else BID_SIDE,
                price = float(price), 
                volume = int(float(volume))
            )
        )
    
    def parse_modify_action(self, line):
        _, side, volume, price, _ = line.split(',')
        self.actions.append(
            Action(
                action_type = MODIFY_ACTION_TYPE,
                side = OFFER_SIDE if side == '1' else BID_SIDE,
                price = float(price), 
                volume = int(float(volume))
            )
        )
    
    def start_new_orderbook(self, line): 
        # periodically clear the order cache so it doesn't eat all the memory 
        if len(self.order_cache) > 50000:
            self.order_cache = {} 
        if self.currBook:
            self.books.append(self.currBook)
        _, _, monotonic_time, exchange_time = line.split(',')
        monotonic_seconds, _, monotonic_nanoseconds = monotonic_time.partition(':')
        epoch_seconds, _, exchange_nanoseconds = exchange_time.partition(':')
        exchange_seconds = long(epoch_seconds) % self.SECONDS_PER_DAY
        self.currBook = OB(
            lastUpdateTime = exchange_seconds * 1000 + int(exchange_nanoseconds) / 1000000.0, 
            lastUpdateMonotonic = int(monotonic_seconds)*1000 + int(monotonic_nanoseconds) / 1000000, 
            actions = self.actions
        )
        self.actions = []

    def unsupported(self, line):
        assert False 
        
    def ignore(self, line):
        pass

    def print_restart(self, line):
        print line
    
    def parse(self, f, debug=False, end=None, drop_out_of_order=False): 
        dispatch_table = {
            'V': self.parse_header, 
            'A': self.parse_add_action, 
            'D': self.parse_delete_action, 
            'M': self.parse_modify_action,
            'O': self.start_new_orderbook, 
            'R': self.print_restart, 
            '#': self.ignore, 
            '\n': self.ignore, 
        }
        try:  
            for line in f:
                start_char = line[0]
                # this loop used to only dispatch on the first char
                # but I inlined the most common function 'build_orderbook_entry'
                # for slight performance boost 
                if start_char != 'Q': 
                    dispatch_table[line[0]](line)
                else: 
                    if line in self.order_cache: 
                        order = self.order_cache[line]
                    else:
                        _, monotonic_timestamp, side, level, price, size, _ =  line.split(',')
                        seconds, _,  nanoseconds = monotonic_timestamp.partition(':')
                        order = Order(
                            timestamp=int(seconds) * 1000 + int(nanoseconds) / 1000000, 
                            side= (side == '1'), 
                            level=int(level),
                            price=float(price),
                            size=int(size)
                        )
                        #if len(self.order_cache)  % 10000 == 0: 
                        #  print len(self.order_cache)
                        self.order_cache[line] = order
                    self.currBook.add_order(order)
        except Exception as inst:
            print "Encountered error at line:", line
            print type(inst)     # the exception instance
            print inst           # __str__ allows args to printed directly
            # sometimes the collector doesn't finish printing 
            # the last orderbook 
            # so skip exceptions at the end of a file 
            if f.read(100) != '':
                print "Unrecoverable!"
                raise
            else: 
                print "At last line of data file, ignoring error..." 
        return self.header, self.books 



def read_books_from_file(f, debug=False, end=None): 
    header = {}
    orderbooks = [] 
    peek_str = f.read(1000)
    f.seek(0)
    # disable the GC since a bug in older python interpreters stupidly scans
    # a list for garbage every time you append to it
    #gc.disable()
    print "Parsing order books..." 
    if peek_str.startswith('V3') or '\nV3' in peek_str:
        parser = V3_Parser()
        header, orderbooks = parser.parse(f, debug, end)
    else:
        orderbooks = books_from_lines_v1(f, debug, end)
        depths = [len(ob.bids) for ob in orderbooks]    
        maxdepth = reduce(max, depths)
        ccy = orderbooks[0].bids[0].ccy
        header = { 'ccy' : ccy }    
    #gc.enable()
    #gc.collect()
    return header, orderbooks 


def read_books(filename, debug=False, end=None): 
    f = open(filename, 'r')
    header, orderbooks = read_books_from_file(f, debug, end) 
    f.close()
    return header, orderbooks 
        
        
parser = OptionParser();
parser.add_option("-r" , "--read", dest="bookfile", help="Read book from file", metavar="FILE")
parser.add_option("-d", "--debug", dest="debug",  action="store_true", help="Print debug output")

if __name__ == '__main__': 
    (options, args) = parser.parse_args()
    if len(sys.argv) == 1:
      print sys.argv[0], "--help for usage\n" 
      exit(-1)
    if options.debug:
      print "Input: ", options.bookfile
    if options.bookfile:
      for x in read_books(options.bookfile, options.debug):
   	    pass
