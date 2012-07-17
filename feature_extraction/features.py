import math 

import numpy as np

def spread(orderBook):
    return orderBook.offers[0].price - orderBook.bids[0].price 
    
## is the market locked or crossed?
def locked(ob):
  return ob.offers[0].price == ob.bids[0].price

def crossed(ob):
  return ob.offers[0].price < ob.bids[0].price


## difference between the best level and 5 levels away

def bid_range(orderBook):
    last_idx = min(4, len(orderBook.bids) - 1)
    return orderBook.bids[0].price - orderBook.bids[last_idx].price
    
def offer_range(orderBook):
    last_idx = min(4, len(orderBook.offers) - 1)
    return orderBook.offers[last_idx].price - orderBook.offers[0].price

def bid_slope(ob):
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

def offer_slope(ob):
    last_idx = min(4, len(ob.offers) - 1)
    first_price = ob.offers[0].price
    cumulative_vol = 0
    total = 0.0
    for i in xrange(last_idx):
      offer = ob.offser[i+1]
      cumulative_vol += offer.size
      delta_p = offer.price - first_price
      total += delta_p / cumulative_vol 
    return total / (last_idx+1) 
  
def best_bid(orderBook):
    return orderBook.bids[0].price

def best_offer(orderBook):
    return orderBook.offers[0].price

def midprice(orderBook):
    return (orderBook.offers[0].price + orderBook.bids[0].price) / 2 

def total_added_volume(ob):
    return ob.added_bid_volume  + ob.added_offer_volume
    
def total_deleted_volume(ob):
    return ob.deleted_bid_volume + ob.deleted_offer_volume

def net_volume(ob):
    return ob.added_bid_volume + ob.added_offer_volume \
      - ob.deleted_bid_volume - ob.deleted_offer_volume


## tr8dr's insertion flow 

def bid_tr8dr(ob):
    return ob.bid_tr8dr

def offer_tr8dr(ob):
    return ob.offer_tr8dr

def tr8dr(ob):
    return ob.bid_tr8dr + offer_tr8dr

## Canceled = deleted from secondary levels 

def canceled_bid_volume(ob):
    return ob.canceled_bid_volume

def canceled_bid_count(ob):
    return ob.canceled_bid_count

def canceled_offer_volume(ob):
    return ob.canceled_offer_volume 

def canceled_offer_count(ob):
    return ob.canceled_offer_count

def total_canceled_volume(ob):
    return ob.canceled_bid_volume + ob.canceled_offer_volume


## Filled = deleted from best level 

def filled_bid_volume(ob):
    return ob.filled_bid_volume

def filled_bid_count(ob):
    return ob.filled_bid_count

def filled_offer_volume(ob):
    return ob.filled_offer_volume

def filled_offer_count(ob):
    return ob.filled_offer_count 

def total_filled_volume(ob):
    return ob.filled_bid_volume  + ob.filled_offer_volume
    


## Add 

def added_offer_volume(ob):
    return ob.added_offer_volume

def added_offer_count(ob):
    return ob.added_offer_count

def added_bid_volume(ob):
    return ob.added_bid_volume

def added_bid_count(ob):
    return ob.added_bid_count



def added_best_offer_volume(ob):
    return ob.added_best_offer_volume

def added_best_offer_count(ob):
    return ob.added_best_offer_count

def added_best_bid_volume(ob):
    return ob.added_best_bid_volume

def added_best_bid_count(ob):
    return ob.added_best_bid_count

## Delete
def deleted_offer_volume(ob):
    return ob.deleted_offer_volume

def deleted_offer_count(ob):
    return ob.deleted_offer_count

def deleted_bid_volume(ob):
    return ob.deleted_bid_volume

def deleted_bid_count(ob):
    return ob.deleted_bid_count


#############

def millisecond_timestamp(orderBook): 
    return orderBook.lastUpdateTime 

def second_timestamp(orderBook):
    return orderBook.lastUpdateTime / 1000.0

def message_count(orderBook):
    return len(orderBook.actions)

#def timestamp(orderBook):
    #return orderBook.lastUpdateTime
    
#def second_timestamp(orderBook):
    #t = orderBook.lastUpdateTime.time()
    #return int(t.hour * 3600 + t.minute * 60 + t.second)

#def millisecond_timestamp(orderBook):
    #t = orderBook.lastUpdateTime.time()
    #return int(t.hour * 3600000 + t.minute * 60000 + t.second * 1000 + int(t.microsecond / 1000.0) )

##def millisecond_timestamp_mod1000(orderBook):
    
#def microsecond_timestamp(orderBook): 
    #t = orderBook.lastUpdateTime.time()
    #return second_timestamp(orderBook)*1000000 + t.microsecond

# cache results 
def bid_volume(orderBook): 
    if hasattr(orderBook, 'bidVolume'): 
        return orderBook.bidVolume 
    else: 
      bidVolume = 0.0 
      for order in orderBook.bids:
          bidVolume += order.size
      orderBook.bidVolume = bidVolume 
      return bidVolume 

# cache results 
def offer_volume(orderBook): 
    if hasattr(orderBook, 'offerVolume'): 
        return orderBook.offerVolume 
    else: 
      offerVolume = 0.0 
      for order in orderBook.offers:
          offerVolume += order.size
      orderBook.offerVolume = offerVolume 
      return offerVolume 

def best_offer_volume(orderBook): 
    return orderBook.offers[0].size

def best_bid_volume(orderBook): 
    return orderBook.bids[0].size 

# volume weighted price of first five bids  
def bid_vwap(ob): 
    p = 0
    v = 0  
    i = 0 
    for order in ob.bids:
        p += order.price * order.size
        v += order.size
        i += 1
        if i >= 4: break  
    return p / v

def offer_vwap(ob):
    p = 0 
    v = 0
    i = 0
    for order in ob.offers:
      p += order.price * order.size 
      v += order.size
      i += 1
      if i >= 4: break 
    return p / v


# average of volume weighted offer and bid prices
def volume_weighted_mid_price(orderBook):
    bidSum = 0 
    for order in orderBook.bids: 
        bidSum += order.price * order.size 
    bidPrice = bidSum / bid_volume(orderBook)

    offerSum = 0 
    for order in orderBook.offers:
        offerSum += order.price * order.size 

    offerPrice = offerSum / offer_volume(orderBook)
    return (offerPrice + bidPrice) / 2 

def prct_best_offer_volume(orderBook): 
    return float(orderBook.offers[0].size) / offer_volume(orderBook)

def prct_best_bid_volume(orderBook): 
    return float(orderBook.bids[0].size) / bid_volume(orderBook)

def log_best_volume_ratio(orderBook):
    offerVol = orderBook.offers[0].size
    bidVol = orderBook.bids[0].size
    if offerVol == 0 or bidVol == 0:
        return 0         
    else:
        return math.log(float(offerVol) / bidVol)

def fraction_of_second(orderBook): 
    t = millisecond_timestamp(orderBook)
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
    
def first_bid_digit(orderBook):
    return nth_digit(orderBook.bids[0].price, 1)

def first_bid_digit_tail(orderBook):
    return nth_digit_tail(orderBook.bids[0].price, 1)
    
def first_offer_digit(orderBook): 
    return nth_digit(orderBook.offers[0].price, 1)

def first_offer_digit_tail(orderBook): 
    return nth_digit_tail(orderBook.offers[0].price, 1)

        
def second_bid_digit(orderBook):
    return nth_digit(orderBook.bids[0].price, 2)
    
def second_bid_digit_tail(orderBook):
    return nth_digit_tail(orderBook.bids[0].price, 2)

def second_offer_digit(orderBook): 
    return nth_digit(orderBook.offers[0].price, 2)

def second_offer_digit_tail(orderBook): 
    return nth_digit_tail(orderBook.offers[0].price, 2)

def third_bid_digit(orderBook):
    return nth_digit(orderBook.bids[0].price, 3)

def third_bid_digit_tail(orderBook):
    return nth_digit_tail(orderBook.bids[0].price, 3)

def third_offer_digit(orderBook): 
    return nth_digit(orderBook.offers[0].price, 3)

def third_offer_digit_tail(orderBook): 
    return nth_digit_tail(orderBook.offers[0].price, 3)

def fourth_bid_digit(orderBook):
    return nth_digit(orderBook.bids[0].price, 4)

def fourth_offer_digit(orderBook): 
    return nth_digit(orderBook.offers[0].price, 4)

def fourth_offer_digit_tail(orderBook): 
    return nth_digit_tail(orderBook.offers[0].price, 4)

def digit_close_to_wrap(d , lim=10):
    mid = lim / 2.0
    dist = abs(mid - d)
    return dist / mid
    
def first_bid_digit_close_to_wrap(orderBook):
    return digit_close_to_wrap(nth_digit_tail(orderBook.bids[0].price, 1), lim =10)
    
def first_offer_digit_close_to_wrap(orderBook):
    return digit_close_to_wrap(nth_digit_tail(orderBook.offers[0].price, 1), lim = 10)

def second_bid_digit_close_to_wrap(orderBook):
    return digit_close_to_wrap(nth_digit_tail(orderBook.bids[0].price, 2), lim = 10)
    
def second_offer_digit_close_to_wrap(orderBook):
    return digit_close_to_wrap(nth_digit_tail(orderBook.offers[0].price, 2), lim = 10)

def third_bid_digit_close_to_wrap(orderBook):
    return digit_close_to_wrap(nth_digit_tail(orderBook.bids[0].price, 3), lim = 10)
    
def third_offer_digit_close_to_wrap(orderBook):
    return digit_close_to_wrap(nth_digit_tail(orderBook.offers[0].price, 3), lim = 10)

def fourth_bid_digit_close_to_wrap(orderBook):
    return digit_close_to_wrap(nth_digit_tail(orderBook.bids[0].price, 4), lim = 10)
    
def fourth_offer_digit_close_to_wrap(orderBook):
    return digit_close_to_wrap(nth_digit_tail(orderBook.offers[0].price, 4), lim = 10)
