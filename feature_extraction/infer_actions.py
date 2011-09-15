import numpy as np 
# heuristics to infer whether changes in the order book reflect an
# add, change or delete, or in the special case of deletions from the 
# first level: a buy or sell 


def buy_volume(ob1, ob2):
    offer1 = ob1.offers[0]
    offer2 = ob2.offers[0]
    if offer1.price < offer2.price:
        return offer1.size
    elif offer1.price == offer2.price and offer1.size > offer2.size:
        return offer1.size - offer2.size
    else: 
        return 0

def sell_volume(ob1, ob2):
    bid1 = ob1.bids[0]
    bid2 = ob2.bids[0]
    if bid1.price > bid2.price: 
        return bid1.size
    elif bid1.price == bid2.price and bid1.size > bid2.size:
        return bid1.size - bid2.size
    else: 
        return 0

def buy_indicator(ob1, ob2):
    offer1 = ob1.offers[0]
    offer2 = ob2.offers[0]
    return (offer1.price < offer2.price) or (offer1.price == offer2.price and offer1.size > offer2.size)
        
def sell_indicator(ob1, ob2):
    bid1 = ob1.bids[0]
    bid2 = ob2.bids[0]
    return (bid1.price > bid2.price) or (bid1.price == bid2.price and bid1.size > bid2.size)

        
def infer_buys_and_sells(orderbooks):
    n = len(orderbooks)
    assert n > 0
    buy_prices = np.zeros(n)
    sell_prices = np.zeros(n)
    buy_sizes = np.zeros(n)
    sell_sizes = np.zeros(n)
    ob1 = orderbooks[0]
    for i in xrange(len(orderbooks) - 1):
        idx = i+1
        ob2 = orderbooks[idx]
        bid1 = ob1.bestBid()
        bid2 = ob2.bestBid()
        if bid1 is not None and bid2 is not None:
            if bid1.price > bid2.price:
                buy_prices[idx] = bid1.price
                buy_sizes[idx] = bid1.size
        
            elif bid1.price == bid2.price and bid1.size > bid2.size:
                buy_prices[idx] = bid1.price
                buy_sizes[idx] = bid1.size - bid2.size
        
        offer1 = ob1.bestOffer()
        offer2 = ob2.bestOffer()
        if offer1 is not None and offer2 is not None:
            if offer1.price < offer2.price:
                sell_prices[idx] = offer1.price
                sell_sizes[idx] = offer1.size
            elif offer1.price == offer2.price and offer1.size > offer2.size:
                sell_prices[idx] = offer1.price
                sell_sizes[idx] = offer1.size - offer2.size
        ob1 = ob2
    return buy_prices, buy_sizes, sell_prices, sell_sizes 
    
    
    
