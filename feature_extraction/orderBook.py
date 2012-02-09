from operator import attrgetter, itemgetter, methodcaller

import orderBookConstants as obc

    
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


OFFER_SIDE = 1 
BID_SIDE = 0

ADD_ACTION_TYPE = 'A'
DELETE_ACTION_TYPE = 'D' 
MODIFY_ACTION_TYPE = 'M'

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
        if not hasattr(self, 'added_volume'):
            self.added_volume = sum([a.volume for a in self.actions if a.action_type == ADD_ACTION_TYPE])
            self.deleted_volume = sum([a.volume for a in self.actions if a.action_type == DELETE_ACTION_TYPE])
            #self.net_volume = self.added_volume - self.deleted_volume
            change_at_price = {}
            for a in self.actions: 
                sign = 1 if a.action_type == ADD_ACTION_TYPE else (-1)
                newval = change_at_price.get(a.price, 0) + sign * a.volume 
                change_at_price[a.price] = newval 
            self.change_at_price = change_at_price
            insertion_flow = 0 
            fill_volume = 0 
            canceled_volume = 0 
            for a in self.actions: 
                if a.action_type == ADD_ACTION_TYPE:
                    if a.side == OFFER_SIDE: 
                        base_price = self.offers[0].price
                        cumulative_volume = sum([order.size for order in self.offers if order.price <= a.price])
                    else: 
                        base_price = self.bids[0].price
                        cumulative_volume = sum([order.size for order in self.bids if order.price >= a.price])
                    prct_change = (a.price - base_price) / base_price 
                    volume_weight = float(a.volume) / (cumulative_volume + a.volume) 
                    insertion_flow += prct_change * volume_weight  
                elif a.action_type == DELETE_ACTION_TYPE:
                    if a.side == OFFER_SIDE:
                        if a.price <= self.offers[0].price: 
                            fill_volume += a.volume 
                        else:
                            canceled_volume += a.volume 
                    else:
                        if a.price >= self.bids[0].price:
                            fill_volume += a.volume
                        else:
                            canceled_volume += a.volume 
            self.fill_volume = fill_volume 
            self.canceled_volume = canceled_volume 
            self.insertion_flow = insertion_flow 
            
