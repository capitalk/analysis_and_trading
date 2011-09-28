import urllib2
import numpy as np

#recorded on july 17th, 2011
old_usd_rates = {
    'cad': .96,
    'aud': .94, 
    'gbp': .62, 
    'chf': .81, 
    'eur': .71, 
    'usd': 1.0, 
    'nzd': 0.78,
    'jpy': 79.06
}

def get_static_usd_rate(currency):
    currency = currency.lower()
    if currency in old_usd_rates:
        return old_usd_rates[currency]
    else: 
        raise RuntimeError("Currency unknown: " + currency)

def get_usd_rate(currency):
    try:
        url ='http://www.multimolti.com/apps/currencyapi/index.php?curr=' + currency 
        return float(urllib2.urlopen(url).readline())
    except: 
        print 'Could not get rate quote from web service, using old currency rates...'
        get_static_usd_rate(currency)
# trade_size_usd = size of each trade, converted from USD to 1st currency in pair
# signal_window_time = how long in the past to look for signals
# min_window_signals = how many signals must agree in a window before we trade 
# min_profit_prct = how much profit must be made before we act on a signal 
# usd_transaction_cost = cost per trade 
def aggressive(ts, bids, offers, signal, currency_pair,  
        trade_size_usd=1000000, 
        signal_window_time=1000, 
        min_window_signals=3, 
        min_profit_prct = 0.0008, 
        usd_transaction_cost=13, 
        min_time_between_trades = 2000, 
        carry_position = False):
    # how much of the payment currency can we buy with 1 dollar? 
    conversion_rate = get_static_usd_rate(currency_pair[1])
    trade_size = trade_size_usd * conversion_rate 
    transaction_cost = usd_transaction_cost * conversion_rate 
    
    BUY = +1
    SELL = -1 
    
    position = 0 
    position_price = 0
    last_trade_time = 0
    
    n = len(ts)
    profits = np.zeros(n)
    position_deltas = np.zeros(n)
    buy_signals_window = []
    sell_signals_window = []
    ignored_sell_signals = 0
    ignored_buy_signals = 0  
    
    action_indices = np.nonzero(signal)[0] 
    for i in action_indices:
        
        t = ts[i] 
        if t - last_trade_time  > min_time_between_trades:
            # clean old signals from window 
            cutoff = t - signal_window_time 
            buy_signals_window = [btime for btime in buy_signals_window if btime >= cutoff]
            sell_signals_window = [stime for stime in sell_signals_window if stime >= cutoff]
                
            curr_signal = signal[i]
            if curr_signal == BUY: buy_signals_window.append(t)
            elif curr_signal == SELL: sell_signals_window.append(t) 
            # should we buy? 
            num_buy_signals = len(buy_signals_window)
            num_sell_signals = len(sell_signals_window)
                
            curr_offer = offers[i]
            curr_bid = bids[i] 
            
            # should we buy? 
            if num_buy_signals > min_window_signals and num_sell_signals == 0:
                if position >= 0:
                    # volume weighted average of old price and new 
                    position_price = (position * position_price + curr_offer * trade_size) / (trade_size + position)
                    position += trade_size
                    position_deltas[i] = trade_size
                    last_trade_time = t 
                    profits[i] = -transaction_cost
                # only buy at a profit if you've previously sold
                elif position < 0: 
                    profit_prct = (position_price - curr_offer) / position_price
                    if min_profit_prct is None or profit_prct >= min_profit_prct:
                        abs_pos = abs(position)
                        position_price = (abs_pos * position_price + curr_offer * trade_size) / (trade_size + abs_pos)
                        position += trade_size
                        position_deltas[i] = trade_size
                        last_trade_time = t 
                        curr_profit = profit_prct * trade_size 
                        profits[i] =  curr_profit - transaction_cost 
                    else:
                        ignored_buy_signals += 1
            # ...or should we sell? 
            elif num_sell_signals > min_window_signals and num_buy_signals == 0:
                if position <= 0:
                    # volume weighted average of old price and new 
                    abs_pos = abs(position)
                    position_price = (abs_pos * position_price + curr_bid * trade_size) / (trade_size + abs_pos)
                    position_deltas[i] = -trade_size
                    position -= trade_size
                    last_trade_time = t 
                    profits[i] = -transaction_cost
                # only sell at a profit if you've previously bought
                elif position > 0: 
                    
                    profit_prct = (curr_bid - position_price) / position_price
                    if min_profit_prct is None or profit_prct >= min_profit_prct:
                        position_price = (position * position_price + curr_offer * trade_size) / (trade_size + position)
                        position_deltas[i] = -trade_size
                        position -= trade_size
                        last_trade_time = t 
                        curr_profit = profit_prct * trade_size 
                        profits[i] =  curr_profit - transaction_cost 
                    else:
                        ignored_sell_signals += 1
    if not carry_position: 
        if position > 0:
            last_bid = np.mean(bids[-100:-5])
            profit = (last_bid - position_price) * trade_size 
            profits[-1] = profit 
            position_deltas[-1] = -position 
        elif position < 0:
            last_offer = np.mean(offers[-100:-5])
            profit = (position_price - last_offer) * trade_size 
            profits[-1] = profit 
            position_deltas[-1] = -position 
            
    usd_profits = profits / conversion_rate 
    usd_position_deltas = position_deltas / conversion_rate 
    usd_last_position = position / conversion_rate 
    return usd_profits, position_deltas, usd_last_position

# signal = +1 to buy, -1 to sell 
# trade_size_usd = size of transactions in USD
# window_time = time in milliseconds to wait for multiple signals
# min_window_signals = min number of consistent trade signals in a window before action 
# slippage = how much worse is the price you get compared with best visible


def aggressive_with_hard_thresholds(ts, bids, offers,  signal, currency_pair,  
        trade_size_usd=1000000, 
        window_time=1000, 
        max_hold_time = 30000, 
        min_window_signals=3, 
        max_loss_prct = 0.0010, 
        min_profit_prct=0.0002, 
        slippage = 0.00001, 
        usd_transaction_cost=13):
    
    # rate from USD to payment currency (ie, for 1 dollar how many yen can I buy?)
    conversion_rate = get_static_usd_rate(currency_pair[1])
    trade_size = trade_size_usd * conversion_rate 
    # convert the transaction cost so they're on the same scale 
    transaction_cost = usd_transaction_cost * conversion_rate 
    
    BUY = +1
    SELL = -1 
    position = 0 
    trade_price = None 
    trade_time = None 
    
    n = len(ts)
    profits = np.zeros(n)
    old_buy_signals = []
    old_sell_signals = [] 
    n = len(bids)
    for i in xrange(n):
        curr_bid = bids[i]
        curr_offer = offers[i]
        curr_signal = signal[i]
        t = ts[i]
        # if have position, check whether need to cut
        # otherwise, if 3 signals in 10ms agree put in an order 
        # enter that position 
        if position > 0:
            slipped_bid = curr_bid * (1  - slippage)
            prct = (slipped_bid - trade_price) / trade_price
            if (prct >= min_profit_prct)  or (prct <= -max_loss_prct) or (t >= trade_time + max_hold_time):
                position = 0 
                profits[i] = prct * trade_size - transaction_cost 
            
        elif position < 0:
            slipped_offer = curr_offer * (1 + slippage)
            prct = (trade_price - slipped_offer) / trade_price 
            if (prct >= min_profit_prct) or (prct <= -max_loss_prct) or (t >= trade_time + max_hold_time):
                position = 0 
                profits[i] = prct * trade_size - transaction_cost  
            
        # no current position, should we enter one? 
        elif curr_signal != 0:
            
            new_buy_signals = []
            new_sell_signals = [] 
            if curr_signal == BUY: new_buy_signals.append(t)
            elif curr_signal == SELL: new_sell_signals.append(t) 
            
            if len(old_buy_signals) > 0 or len(old_sell_signals) > 0:
                cutoff = t - window_time 
                for btime in old_buy_signals:
                    if btime >= cutoff: new_buy_signals.append(btime)
                for stime in old_sell_signals:
                    if stime >= cutoff: new_sell_signals.append(stime)
            # should we buy? 
            nbuys = len(new_buy_signals)
            nsells = len(new_sell_signals)
            if nbuys > min_window_signals and nsells == 0:
                # assume we can always buy at this size! 
                position = trade_size 
                trade_price = curr_offer  * (1-slippage)
                trade_time = t 
                old_buy_signals = [] 
                old_sell_signals = []
                profits[i] -= transaction_cost
                # ...or should we sell? 
            elif nsells > min_window_signals and nbuys == 0:
                # assume we can always sell at this size! 
                position = -trade_size 
                trade_price = curr_bid * (1+slippage)
                trade_time = t 
                old_buy_signals = [] 
                old_sell_signals = []
                profits[i] -= transaction_cost 
                # ...or should we do nothing? 
            else:
                old_buy_signals = new_buy_signals
                old_sell_signals = new_sell_signals 
                
    # what if we're still holding a position at the end of the day?
    if position > 0:
        slipped_bid = curr_bid * (1  - slippage)
        prct = (slipped_bid - trade_price) / trade_price
        profits[n-1] += prct * trade_size - transaction_cost 
    elif position < 0:
        slipped_offer = curr_offer * (1 + slippage)
        prct = (trade_price - slipped_offer) / trade_price 
        profits[n-1] += prct * trade_size - transaction_cost  
        
    usd_profits = profits / conversion_rate 
    return usd_profits 

# wrapper for simulate profits so we don't have to explicitly pass bids, offers, times 
#def aggressive_with_hard_thresholds_dataset(dataset, signal, start_index=0):
#    bids = dataset['bid/100ms'][start_index:]
#    offers = dataset['offer/100ms'][start_index:]
#    ts = dataset['t'][start_index:]
#    return simulate_profits(ts, bids, offers, signal, d.currency_pair)
        
    
def profit_vs_threshold(dataset, signal): 
    thresholds = np.arange(0.00005, 0.01, 0.00005)
    num_thresholds = len(thresholds)
    import progressbar 
    progress = progressbar.ProgressBar(num_thresholds).start()

    cumulative_profits = []
    for i in xrange(num_thresholds):
        t = thresholds[i] 
        profits = aggressive_with_hard_thresholds_dataset(dataset, signal, min_profit_prct = t)
        cumulative_profits.append(sum(profits))
        progress.update(i)
    progress.finish()
    import pylab
    pylab.plot(thresholds, cumulative_profits)
    pylab.xlabel('take-profit threshold as percent of original price')
    pylab.ylabel('profit at end of day')
    pylab.title('take-profit threshold vs profit')
    return thresholds, cumulative_profits
