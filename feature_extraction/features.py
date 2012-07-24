import math 

def spread(ob, stats):
  return stats.best_offer_price - stats.best_bid_price
    
## is the market locked or crossed?
def locked(ob, stats):
  return stats.best_offer_price == stats.best_bid_price

def crossed(ob, stats):
  return stats.best_offer_price < stats.best_bid_price
  
## difference between the best level and 5 levels away

def bid_range(ob, stats):
    last_idx = min(4, len(ob.bids) - 1)
    return ob.bids[0].price - ob.bids[last_idx].price
    
def offer_range(ob, stats):
    last_idx = min(4, len(ob.offers) - 1)
    return ob.offers[last_idx].price - ob.offers[0].price

def bid_slope(ob, stats):
    last_idx = min(4, len(ob.bids) - 1)
    first_price = ob.bids[0].price
    cumulative_vol = 0
    total = 0.0
    for i in xrange(last_idx):
      bid = ob.bids[i+1]
      cumulative_vol += bid.size
      delta_p = bid.price - first_price
      total += delta_p / cumulative_vol 
    return total / (last_idx+1)   

def offer_slope(ob, stats):
    last_idx = min(4, len(ob.offers) - 1)
    first_price = ob.offers[0].price
    cumulative_vol = 0
    total = 0.0
    for i in xrange(last_idx):
      offer = ob.offers[i+1]
      cumulative_vol += offer.size
      delta_p = offer.price - first_price
      total += delta_p / cumulative_vol 
    return total / (last_idx+1) 
  
def best_bid(orderBook, stats):
    return stats.best_bid_price 

def best_offer(orderBook, stats):
    return stats.best_offer_price 
    
def midprice(orderBook, stats):
    return (stats.best_offer_price + stats.best_bid_price ) / 2 

def total_added_volume(ob, stats):
    return ob.stats.added_bid_volume  + ob.stats.added_offer_volume


def deleted_bid_volume(ob, stats):
    return stats.deleted_bid_volume

def deleted_offer_volume(ob, stats):
    return stats.deleted_offer_volume
        
def total_deleted_volume(ob, stats):
    return stats.deleted_offer_volume + stats.deleted_bid_volume

def net_volume(ob, stats):
    return ob.added_bid_volume + ob.added_offer_volume \
      - deleted_bid_volume(ob) - deleted_offer_volume(ob)
      

## tr8dr's insertion flow 

def bid_tr8dr(ob, stats):
    return stats.bid_tr8dr

def offer_tr8dr(ob, stats):
    return stats.offer_tr8dr

def tr8dr(ob, stats):
    return stats.bid_tr8dr + stats.offer_tr8dr

## Canceled = deleted from secondary levels 

def canceled_bid_volume(ob, stats):
    return stats.canceled_bid_volume

def canceled_bid_count(ob, stats):
    return stats.canceled_bid_count

def canceled_offer_volume(ob, stats):
    return stats.canceled_offer_volume 

def canceled_offer_count(ob, stats):
    return stats.canceled_offer_count

def total_canceled_volume(ob, stats):
    return stats.canceled_bid_volume + stats.canceled_offer_volume


## Filled = deleted from best level 

def filled_bid_volume(ob, stats):
    return stats.filled_bid_volume

def filled_bid_count(ob, stats):
    return stats.filled_bid_count

def filled_offer_volume(ob, stats):
    return stats.filled_offer_volume

def filled_offer_count(ob, stats):
    return stats.filled_offer_count 

def total_filled_volume(ob, stats):
    return stats.filled_bid_volume  + stats.filled_offer_volume
    
## Add 

def added_offer_volume(ob, stats):
    return stats.added_offer_volume

def added_offer_count(ob, stats):
    return stats.added_offer_count

def added_bid_volume(ob, stats):
    return stats.added_bid_volume

def added_bid_count(ob, stats):
    return stats.added_bid_count



def added_best_offer_volume(ob, stats):
    return stats.added_best_offer_volume

def added_best_offer_count(ob, stats):
    return stats.added_best_offer_count

def added_best_bid_volume(ob, stats):
    return stats.added_best_bid_volume

def added_best_bid_count(ob, stats):
    return stats.added_best_bid_count

## Delete
def deleted_offer_volume(ob, stats):
    return stats.deleted_offer_volume

def deleted_offer_count(ob, stats):
    return stats.deleted_offer_count

def deleted_bid_volume(ob, stats):
    return stats.deleted_bid_volume

def deleted_bid_count(ob, stats):
    return stats.deleted_bid_count


#############

def millisecond_timestamp(orderBook, stats): 
    return orderBook.lastUpdateTime 

def second_timestamp(orderBook, stats):
    return orderBook.lastUpdateTime / 1000.0

def message_count(orderBook, stats):
    return len(orderBook.actions)

# cache results 
def bid_volume(orderBook, stats): 
    if hasattr(orderBook, 'bidVolume'): 
        return orderBook.bidVolume 
    else: 
      bidVolume = 0.0 
      for order in orderBook.bids:
          bidVolume += order.size
      orderBook.bidVolume = bidVolume 
      return bidVolume 

# cache results 
def offer_volume(orderBook, stats): 
    if hasattr(orderBook, 'offerVolume'): 
        return orderBook.offerVolume 
    else: 
      offerVolume = 0.0 
      for order in orderBook.offers:
          offerVolume += order.size
      orderBook.offerVolume = offerVolume 
      return offerVolume 

def best_offer_volume(ob, stats): 
    return stats.best_offer_volume

def best_bid_volume(ob, stats): 
    return stats.best_bid_volume
    
# volume weighted price of first five bids  
def bid_vwap(ob, stats): 
    p = 0
    v = 0  
    i = 0 
    for order in ob.bids:
        p += order.price * order.size
        v += order.size
        i += 1
        if i >= 4: break  
    return p / v

def offer_vwap(ob, stats):
    p = 0 
    v = 0
    i = 0
    for order in ob.offers:
      p += order.price * order.size 
      v += order.size
      i += 1
      if i >= 4: break 
    return p / v

def fraction_of_second(orderBook, stats): 
    t = orderBook.lastUpdateTime 
    return (t % 1000.0) / 1000.0

def nth_digit(x,n):
    x = abs(x)
    int_x = int(x) 
    if int_x != 0: x -= int_x 
    if x == 0: return 0 
    else:
        # add very small constant to keep from rounding down 
        return int( 0.000001 + 
            x / (10 ** int(math.log10(x) - n))
            ) % 10
    
# like nth digit, but don't floor at the end.
# so nth_digit_tail(3.14,1) = 1.4 
def nth_digit_tail(x,n):
    x = abs(x)
    int_x = int(x) 
    if int_x != 0: x -= int_x
    if x == 0: return 0
    else: return (x / (10 ** int(math.log10(x) - n))) % 10 
    

