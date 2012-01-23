import math 


def spread(orderBook):
    return orderBook.offers[0].price - orderBook.bids[0].price 
    
def bid_range(orderBook):
    return orderBook.bids[0].price - orderBook.bids[-1].price
    
def offer_range(orderBook):
    return orderBook.offers[-1].price - orderBook.offers[0].price

def best_bid(orderBook):
    return orderBook.bids[0].price

def best_offer(orderBook):
    return orderBook.offers[0].price

def midprice(orderBook):
    return (orderBook.offers[0].price + orderBook.bids[0].price) / 2 

def total_added_volume(ob):
    ob.compute_order_flow_stats()
    return ob.added_volume 
    
def total_deleted_volume(ob):
    ob.compute_order_flow_stats()
    return ob.deleted_volume

def net_volume(ob):
    ob.compute_order_flow_stats()
    return ob.added_volume - ob.deleted_volume

def canceled_volume(ob):
    ob.compute_order_flow_stats()
    return ob.canceled_volume 
    
def fill_volume(ob):
    ob.compute_order_flow_stats()
    return ob.fill_volume 
    
def insertion_flow(ob):
    ob.compute_order_flow_stats()
    return ob.insertion_flow

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

# in millions 
def best_offer_volume(orderBook): 
    return float(orderBook.offers[0].size) / 1000000

# in millions
def best_bid_volume(orderBook): 
    return float(orderBook.bids[0].size) / 1000000 

# prices from either side weighted by volume 
def volume_weighted_overall_price(orderBook): 
    p = 0 
    for order in orderBook.bids: 
        p += order.price * order.size 
    for order in orderBook.offers:
        p += order.price * order.size 
    return p / (bid_volume(orderBook) + offer_volume(orderBook))


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

def spread(orderBook):
    return best_offer(orderBook) - best_bid(orderBook) 

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

def fourth_bid_digit(orderBook):
    return nth_digit_tail(orderBook.bids[0].price, 4)

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
