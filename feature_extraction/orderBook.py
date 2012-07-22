
from collections import namedtuple 
  
OFFER_SIDE = True
BID_SIDE = False
  
class Order:
    def __init__(self, 
            timestamp=None,
            side=None, 
            level=None, 
            price=None, 
            size=None):  # orderdepthcount=None, ccy = None
        self.timestamp = timestamp
        self.side = side
        self.level = level
        self.price = price
        self.size = size
        #self.orderdepthcount = orderdepthcount
        #self.ccy = ccy 
    
    def __str__(self):
      return "%s, %s, %s, %s, %s" % (self.timestamp, self.level, self.price, self.size)

    def p(self):
        print self.__str__()


ADD_ACTION_TYPE = 'A'
DELETE_ACTION_TYPE = 'D' 
MODIFY_ACTION_TYPE = 'M'

# actions are now just tuples, so use these positions instead of fields
Action = namedtuple('Action', ('action_type', 'side', 'price', 'size'))

class OB:
    __slots__ = \
      ['bids', 'offers', 'day', 'lastUpdateTime', 'lastUpdateMonotonic', 
       'actions', 
       
       'bid_tr8dr', 'offer_tr8dr', 
       
       'filled_bid_volume', 'filled_bid_count', 
       'filled_offer_volume', 'filled_offer_count',
       'deleted_bid_volume', 'deleted_bid_count',
       'deleted_offer_volume', 'deleted_offer_count',
       'canceled_bid_volume', 'canceled_bid_count',
       'canceled_offer_volume', 'canceled_offer_count',
       'added_bid_volume', 'added_bid_count',  
       'added_offer_volume', 'added_offer_count',
       'added_best_bid_volume', 'added_best_bid_count', 
       'added_best_offer_volume', 'added_best_offer_count']
    
    """day = days since unix start time"""
    def __init__(self, day, lastUpdateTime, lastUpdateMonotonic = None, actions = []):
        self.bids = [] 
        self.offers = [] 
        self.day = day
        self.lastUpdateTime = lastUpdateTime
        self.lastUpdateMonotonic = lastUpdateMonotonic
        self.actions = actions 
        
        self.bid_tr8dr = 0 
        self.offer_tr8dr = 0

        self.filled_bid_volume = 0
        self.filled_bid_count = 0 
        self.filled_offer_volume = 0
        self.filled_offer_count = 0

        self.canceled_bid_volume = 0 
        self.canceled_bid_count = 0
        self.canceled_offer_volume = 0
        self.canceled_offer_count = 0

        self.added_offer_volume = 0
        self.added_offer_count = 0
        self.added_bid_volume = 0
        self.added_bid_count = 0

        self.added_best_offer_volume = 0 
        self.added_best_offer_count = 0
        self.added_best_bid_volume = 0
        self.added_best_bid_count = 0
    
    def add_order(self, order):
        if order.side == OFFER_SIDE: 
            self.offers.append(order)
        else:
            self.bids.append(order) 
        
    def add_bid(self, order):
        #assert len(self.bids) == 0 or entry.price < self.bids[-1].price
        self.bids.append(order)

    def add_offer(self, order):
        #assert len(self.offers) == 0 or entry.price > self.offers[-1].price
        self.offers.append(order) 
    
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
        
    def compute_order_flow_stats(self): 
      best_offer_price = self.offers[0].price
      best_bid_price = self.bids[0].price
 
      for a in self.actions: 
        action_type, side, p, v  = a
        if action_type == ADD_ACTION_TYPE:
          self.offer_tr8dr += (p - best_offer_price) / best_offer_price 
          if side == OFFER_SIDE:
            self.added_offer_volume += v
            self.added_offer_count += 1
            if p <= best_offer_price:
              self.added_best_offer_volume += v
              self.added_best_offer_count += 1
                       
            #base_price = best_offer_price
            #cumulative_volume = sum([order.size for order in self.offers if order.price <= p])
          else: 
            self.bid_tr8dr += (p - best_bid_price) / best_bid_price 
            self.added_bid_volume += v
            self.added_bid_count += 1
            if p >= best_bid_price:
              self.added_best_bid_volume += v
              self.added_best_bid_count += 1
            #base_price = best_bid_price
            #cumulative_volume = sum([order.size for order in self.bids if order.price >= p])
        elif action_type == DELETE_ACTION_TYPE:
          if side == OFFER_SIDE:
            self.offer_tr8dr -= (p - best_offer_price) / best_offer_price 
            if p <= best_offer_price: 
              self.filled_offer_volume += v 
              self.filled_offer_count += 1 
            else:
              self.canceled_offer_volume += v
              self.canceled_offer_count += 1
          else:
            self.bid_tr8dr -= (p - best_bid_price) / best_bid_price 
            if p >= best_bid_price:
              self.filled_bid_volume += v
              self.filled_bid_count += 1
            else:
              self.canceled_bid_volume += v
              self.canceled_bid_count += 1
      self.deleted_bid_volume = self.canceled_bid_volume + self.filled_bid_volume
      self.deleted_bid_count = self.canceled_bid_count + self.filled_bid_count
      self.deleted_offer_volume = self.canceled_offer_volume + self.filled_offer_volume 
      self.deleted_offer_count = self.canceled_offer_count + self.filled_offer_count
         
