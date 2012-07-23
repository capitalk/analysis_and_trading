#!/usr/bin/python
from optparse import OptionParser
from datetime import datetime

import sys
from orderBook import Order, OB
from orderBook import ADD_ACTION_TYPE, MODIFY_ACTION_TYPE, DELETE_ACTION_TYPE
from orderBook import BID_SIDE, OFFER_SIDE 
import orderBookConstants


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

def parse_ccy(ccy_str):
  if len(ccy_str) == 6:
    ccy1 = ccy_str[:3]
    ccy2 = ccy_str[3:6]
  elif len(ccy_str) == 7:
    # if it's not 6 characters it must have a separator 
    sep = ccy_str[3]
    if sep != '/' and sep != '_' and sep != "-":
      raise RuntimeError("Unrecognized currency pair separator " + sep + " in " + ccy_str)
    ccy1 = ccy_str[:3]
    ccy2 = ccy_str[4:7]
  else:
    raise RuntimeError("Unrecognize currency pair format: " + ccy_str)
  return ccy1, ccy2 

def parse_filename(filename):
  base = filename.split("/")[-1].split(".")[0]
  fields = base.split("_")
  assert len(fields) >= 5
  venue, ccy_str, year, month, day = fields[:5]
  ccy_pair = parse_ccy(ccy_str)
  return { 
    'ccy': ccy_pair, 
    'venue': venue.lower(),
    'year': int(year),
    'month': int(month),
    'day': int(day)
  }

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


class V3_Parser:
    def __init__(self): 
        self.header = {} 
        # sometimes the files are missing their first line, 
        # in which case we need to reconstruct the header info 
        self.done_with_header = False 
        self.currBook = None
        self.books = [] 
        # keep this around for printing debug info on restarts
        self.actions = []    
        self.SECONDS_PER_DAY = 60 * 60 * 24     
        self.order_cache = {}


    def header_ok(self, f):
      old_pos = f.tell()
      line = f.readline()
      while len(line) == 0:
        line = f.readline()
      # return the position so we leave no externally visible changes
  
      if line.startswith('V3'):
          v, ccy, _  = line.split(',')
          f.seek(old_pos)
          return v == 'V3'
      elif line.startswith('RESTART'):
        try:
          while not line.startswith('OB'):
            line = f.readline()
          # found an order book, so file is probably OK
          f.seek(old_pos)
          return True
        except StopIteration:
          f.seek(old_pos)
          return False
      else:
        f.seek(old_pos)
        return False 
      
           
    def parse_header(self, line): 
      # make sure nothing else has been parsed yet 
      assert self.currBook == None 
      assert len(self.books) == 0 
      assert self.actions == [] 
      assert len(self.header) == 0
      fields = line.split(',')
      assert len(fields) == 3
      v, ccy_str, _ = fields
      assert v == 'V3'
      self.header['ccy'] = parse_ccy(ccy_str)
      self.done_with_header = True 
          
    def parse_add_action(self, line):
        _, side, volume, price, _ = line.split(',')
        action_type = ADD_ACTION_TYPE,
        side = OFFER_SIDE if side == '1' else BID_SIDE,
        price = float(price), 
        volume = int(float(volume))
        self.actions.append( (action_type, side, price, volume) )
            
    
    def parse_delete_action(self, line):
        _, side, volume, price, _ = line.split(',')
        action_type = DELETE_ACTION_TYPE
        side = OFFER_SIDE if side == '1' else BID_SIDE,
        price = float(price)
        volume = int(float(volume))
        self.actions.append( (action_type, side, price, volume) )
          
    def parse_modify_action(self, line):
        _, side, volume, price, _ = line.split(',')
        action_type = MODIFY_ACTION_TYPE
        side = OFFER_SIDE if side == '1' else BID_SIDE
        price = float(price)
        volume = int(float(volume))
        self.actions.append( (action_type, side, price, volume) )
            
    def start_new_orderbook(self, line):         
        self.at_start_of_file = False 
        # periodically clear the order cache so it doesn't eat all the memory 
        if len(self.order_cache) > 5000:
            self.order_cache.clear() 

        if self.currBook:
            self.books.append(self.currBook)
        _, _, monotonic_time, exchange_time = line.split(',')
        monotonic_seconds, _, monotonic_nanoseconds = monotonic_time.partition(':')
        epoch_seconds, _, exchange_nanoseconds = exchange_time.partition(':')
        epoch_seconds = long(epoch_seconds)
        exchange_seconds =  epoch_seconds % self.SECONDS_PER_DAY
        exchange_day = epoch_seconds / self.SECONDS_PER_DAY
        update_time = exchange_seconds * 1000 + long(exchange_nanoseconds) / (10**6)
        monotonic_time = long(monotonic_seconds)*1000 + long(monotonic_nanoseconds) / (10**6)
        self.currBook = OB(
            day = exchange_day, 
            lastUpdateTime = update_time, 
            lastUpdateMonotonic = monotonic_time, 
            actions = self.actions
        )
        self.actions = []

    def unsupported(self, line):
        assert False 
        
    def ignore(self, line):
        pass

 
    def parse(self, f, debug=False, end=None, drop_out_of_order=False, start_from_line = None):
      
         
        dispatch_table = {
            'V': self.parse_header, 
            'A': self.parse_add_action, 
            'D': self.parse_delete_action, 
            'M': self.parse_modify_action,
            'O': self.start_new_orderbook, 
            '#': self.ignore, 
            '\n': self.ignore, 
        }
        for line in f:
          try:
              # this loop used to only dispatch on the first char
              # but I inlined the most common function 'build_orderbook_entry'
              # for slight performance boost 
              if line[0] == 'Q':  
                  if line in self.order_cache: 
                      order = self.order_cache[line]
                  else:
                      _, monotonic_timestamp, side, level, price, size, _ =  line.split(',')
                      seconds, _,  nanoseconds = monotonic_timestamp.partition(':')
                      order = Order(
                          timestamp = int(seconds) * 1000 + int(nanoseconds) / 1000000, 
                          side = (side == '1'), 
                          level = int(level),
                          price = float(price),
                          size = int(size)
                      )
                      self.order_cache[line] = order
                  self.currBook.add_order(order)
              else:
                dispatch_table[line[0]](line) 
           
          except Exception as inst:     
            # sometimes the collector doesn't finish printing 
            # the last orderbook 
            # so skip exceptions at the end of a file 
            curr_pos = f.tell()
            peek_str = f.read(100)
            f.seek(curr_pos)
            if peek_str == '' and len(self.books) > 0:
                print "At last line of data file, ignoring error..." 
                break
            elif "RESTART" in line:
                print "Found restart without preceding newline"
                print ">> ", line
                # had to inline restarts since, if they happen at the 
                # beginning of the file, they force us to wind forward to
                # the first orderbook 
                if self.done_with_header:
                  continue
                else:
                  assert self.currBook == None
                  assert len(self.books) == 0 
                  assert len(self.header) == 0
                  # if we haven't parsed a header yet then skip to first
                  # orderbook we can find 
                  line = next(f)
                  while not line.startswith('OB'):
                    line = next(f)
                  fields = line.split(',')
                  assert len(fields) >= 2
                  ccy_str = fields[1]
                  self.header['ccy'] = parse_ccy(ccy_str)
                  # call the normal routine to start an orderbook
                  self.start_new_orderbook(line)
                  self.done_with_header = True
                  continue
            else: 
                print "Encountered error at line:", line
                print type(inst)     # the exception instance
                print inst           # __str__ allows args to printed directly
                print "Unrecoverable!"
                raise
                
        return self.header, self.books 


import gzip 
def open_gzip(filename):
  return gzip.GzipFile(filename, 'r')

def read_books_from_filename(filename, debug=False, end=None): 
  
    print "Parsing order books..."
    
    # since our file format still doesn't put enough metadata 
    # at in its header, we rely on the filename to give us the venue 
    # and date 
    inferred_header = parse_filename(filename)
    
    if filename.endswith('.gz'):
        f = open_gzip(filename)
    else:
        f = open(filename, 'r')
    
    v3_parser = V3_Parser() 
    
    if v3_parser.header_ok(f):
        v3_header, orderbooks = v3_parser.parse(f, debug, end)
        assert inferred_header['ccy'] == v3_header['ccy'] 
    else:
        orderbooks = books_from_lines_v1(f, debug, end)
        depths = [len(ob.bids) for ob in orderbooks]    
        maxdepth = reduce(max, depths)
        ccy = orderbooks[0].bids[0].ccy
        assert ccy == inferred_header['ccy']  
    f.close()
    return inferred_header, orderbooks 


#def read_books(filename, debug=False, end=None): 
#    f = open(filename, 'r')
#    header, orderbooks = read_books_from_file(f, debug, end) 
#    f.close()
#    return header, orderbooks 
        
        
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
      for x in read_books_from_filename(options.bookfile, options.debug):
   	    pass
