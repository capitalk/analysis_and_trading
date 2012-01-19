from operator import attrgetter, itemgetter, methodcaller

import orderBookConstants as obc

    
class Order:
    def __init__(self, 
            timestamp=None, 
            side=None, 
            level=None, 
            price=None, 
            size=None, 
            orderdepthcount=None, 
            ccy = None):
        self.timestamp = timestamp
        self.side = side
        self.level = level
        self.price = price
        self.size = size
        self.orderdepthcount = orderdepthcount
        self.ccy = ccy 
    
    def __str__(self):
      return "%s, %s, %s, %s, %s, %s" % (self.timestamp, self.side, self.level, self.price, self.size, self.orderdepthcount)

    def p(self):
        print self.__str__()

class Action:
    def __init__(self, action_type = None, side = None, price = None, volume = None):
        self.action_type = action_type 
        self.side = side
        self.price = price
        self.volume = volume

class OB:
    def __init__(self, lastUpdateTime, lastUpdateMonotonic = None, actions = []):
        self.bids = [] 
        self.offers = [] 
        self.lastUpdateTime = lastUpdateTime
        self.lastUpdateMonotonic = lastUpdateMonotonic
        self.actions = actions 
        self.side_lookup = {0:self.bids, 1:self.offers}
    
    def add_order(self, order):
        self.side_lookup[order.side].append(order)
        if (self.lastUpdateTime is None) or (self.lastUpdateTime < order.timestamp): 
            self.lastUpdateTime = order.timestamp 

    def add_bid(self, entry):
        #assert len(self.bids) == 0 or entry.price < self.bids[-1].price
        self.bids.append(entry)
        if (self.lastUpdateTime is None) or (self.lastUpdateTime < entry.timestamp): 
            self.lastUpdateTime = entry.timestamp 
    def add_offer(self, entry):
        #assert len(self.offers) == 0 or entry.price > self.offers[-1].price
        self.offers.append(entry) 
        if (self.lastUpdateTime is None) or (self.lastUpdateTime < entry.timestamp): 
            self.lastUpdateTime = entry.timestamp 
    
    def best_bid(self): 
        if len(self.bids) > 0:
            return self.bids[0] 
        else:
            return None

    def best_offer(self): 
        if len(self.offers) > 0:
            return self.offers[0] 
        else:
            return None
    def __str__(self): 
      s = ""    
      for order in reversed(self.offers):
        s += "ASK: "
        s += str(order) + "\n" 
      s += "<------- SPREAD \n"
      for order in self.bids:
        s+= "BID: " + str(order) + "\n" 
      return s

    def p(self):
        print self.__str__()

