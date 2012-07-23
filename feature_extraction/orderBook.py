
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

ob_stat_names = (
 'bid_tr8dr', 'offer_tr8dr', 
 'filled_bid_volume', 'filled_bid_count', 
 'filled_offer_volume', 'filled_offer_count',
 'canceled_bid_volume', 'canceled_bid_count',
 'canceled_offer_volume', 'canceled_offer_count',
 'added_bid_volume', 'added_bid_count',  
 'added_offer_volume', 'added_offer_count',
 'added_best_bid_volume', 'added_best_bid_count', 
 'added_best_offer_volume', 'added_best_offer_count', 
 'total_bid_volume', 'total_offer_volume' )
 
 # these are redundant since deleted = filled + canceled
 #'deleted_bid_volume', 'deleted_bid_count',
 #'deleted_offer_volume', 'deleted_offer_count',


ob_stat_init_zeros = [0] * len(ob_stat_names)

OrderBookStats = namedtuple('OrderBookStats', ob_stat_names)

class OB:
    __slots__ = [
      'bids', 'offers', 'day', 'lastUpdateTime', 
      'lastUpdateMonotonic', 'actions', 'stats']
      
    """day = days since unix start time"""
    def __init__(self, day, lastUpdateTime, lastUpdateMonotonic = None, actions = []):
        self.bids = [] 
        self.offers = [] 
        self.day = day
        self.lastUpdateTime = lastUpdateTime
        self.lastUpdateMonotonic = lastUpdateMonotonic
        self.actions = actions 
        self.stats = OrderBookStats(*ob_stat_init_zeros)
        
    def add_order(self, order):
      if order.side == OFFER_SIDE: 
        self.offers.append(order)
      else:
        self.bids.append(order) 
        
    def add_bid(self, order):
      self.bids.append(order)

    def add_offer(self, order):
      self.offers.append(order) 
    
            
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
        
    def compute_stats(self): 
      best_offer_price = self.offers[0].price
      best_bid_price = self.bids[0].price
      stats = self.stats 
      
      for a in self.actions: 
        action_type, side, p, v  = a
        if action_type == ADD_ACTION_TYPE:
          stats.offer_tr8dr += (p - best_offer_price) / best_offer_price 
          if side == OFFER_SIDE:
            stats.added_offer_volume += v
            stats.added_offer_count += 1
            if p <= best_offer_price:
              stats.added_best_offer_volume += v
              stats.added_best_offer_count += 1
                       
            #base_price = best_offer_price
            #cumulative_volume = sum([order.size for order in self.offers if order.price <= p])
          else: 
            stats.bid_tr8dr += (p - best_bid_price) / best_bid_price 
            stats.added_bid_volume += v
            stats.added_bid_count += 1
            if p >= best_bid_price:
              stats.added_best_bid_volume += v
              stats.added_best_bid_count += 1
            #base_price = best_bid_price
            #cumulative_volume = sum([order.size for order in self.bids if order.price >= p])
        elif action_type == DELETE_ACTION_TYPE:
          if side == OFFER_SIDE:
            stats.offer_tr8dr -= (p - best_offer_price) / best_offer_price 
            if p <= best_offer_price: 
              stats.filled_offer_volume += v 
              stats.filled_offer_count += 1 
            else:
              stats.canceled_offer_volume += v
              stats.canceled_offer_count += 1
          else:
            stats.bid_tr8dr -= (p - best_bid_price) / best_bid_price 
            if p >= best_bid_price:
              stats.filled_bid_volume += v
              stats.filled_bid_count += 1
            else:
              stats.canceled_bid_volume += v
              stats.canceled_bid_count += 1
      #stats.deleted_bid_volume = stats.canceled_bid_volume + stats.filled_bid_volume
      #stats.deleted_bid_count = stats.canceled_bid_count + stats.filled_bid_count
      #stats.deleted_offer_volume = stats.canceled_offer_volume + stats.filled_offer_volume 
      #stats.deleted_offer_count = stats.canceled_offer_count + stats.filled_offer_count