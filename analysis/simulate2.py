import numpy as np
import scipy.stats as stats
import logging as log
import mm_utils
import random
import time

#recorded on july 17th, 2011
old_usd_rates = {
    'cad': .96,
    'aud': .94, 
    'gbp': .62, 
    'chf': .81, 
    'eur': .71, 
    'usd': 1.00, 
    'nzd': 0.78,
    'jpy': 79.06
}

def get_cross_rate(currency):
    currency = currency.lower()
    if currency in old_usd_rates:
        return old_usd_rates[currency]
    else: 
        raise RuntimeError("Currency unknown: " + currency)

def trade_stats(signals, usd_profits, ts):
    non_zero_signals = np.nonzero(signals)
    signal_count = np.count_nonzero(signals)
    buys = [signals[x] == 1 for x in non_zero_signals]
    sells = [signals[x] == -1 for x in non_zero_signals]
    #buy_count = np.count_nonzero(buys)
    #sell_count = np.count_nonzero(sells)
    
    trade_by_trade = [usd_profits[x] for x in non_zero_signals]
     
    print "---------------------------------------------------------"
    (h,m,s,ms) = mm_utils.millis_to_hmsms(ts[-1])
    duration_str = "%d hours, %d minutes, %d seconds, %d ms" % (h, m, s, ms)
    print "Evaluatiing: ", "[", ts[0], "-", ts[-1], "]", duration_str 
    print "Signal count: ", signal_count
    print "PnL (USD): ", sum(usd_profits)
    print "Buy signals: ", np.count_nonzero(buys)
    print "Sell signals: ", np.count_nonzero(sells)
    print "--------------------Ignored Signals----------------------"
    pos_limit = [x for x in ignored_signals if abs(x) == 2]
    long_pos_limit = [x for x in ignored_signals if x == 2]
    short_pos_limit = [x for x in ignored_signals if x == -2]
    print "Position limit hit:", pos_limit, " long: ", long_pos_limit, " short: ", short_pos_limit 

    min_pnl = [x for x in ignored_signals if abs(x) == 3]
    long_min_pnl = [x for x in ignored_signals if x == 3]
    short_min_pnl = [x for x in ignored_signals if x == -3]
    print "Min pnl missed:", min_pnl, " long: ", long_min_pnl, " short: ", short_min_pnl 

    min_time = [x for x in ignored_signals if abs(x) == 4]
    long_min_time = [x for x in ignored_signals if x == 4]
    short_min_time = [x for x in ignored_signals if x == -4]
    print "Min time between trades:", min_time, " long: ", long_min_time, " short: ", short_min_time 
    print "---------------------------------------------------------"
    #return (signal_count, buys, sells, trade_by_trade)

def fill_binomial(n=1, p=0.8):
   return stats.binom.ppf(np.random.rand(), n, p) 

# trade_size_usd = size of each trade, converted from USD to 1st currency in pair
# signal_window_time = how long in the past to look for signals
# min_window_signals = how many signals must agree in a window before we trade 
# min_profit_prct = how much profit must be made before we act on a signal 
# usd_transaction_cost = cost per trade 
def execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signal, currency_pair,  
        trade_size_usd=1000000, 
        signal_window_time=200, 
        min_window_signals=3, 
        min_profit_prct = 0.0008, 
        usd_transaction_cost=13, 
        min_time_between_trades=1000, 
        carry_position = True,
        max_position = None, 
        trade_size_scalar=2,
        fill_function=fill_binomial,
        LOG=False):

    # Setup logging if needed
    if LOG == True:
        log_filename = "SIMLOG."+currency_pair[0]+currency_pair[1]+".log" 

        # create logger
        l = log.getLogger(__name__)
        l.setLevel(log.DEBUG)
        # create file handler and set level to debug
        ch = log.FileHandler(log_filename, 'w')
        ch.setLevel(log.DEBUG)
        # create formatter
        formatter = log.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        l.addHandler(ch)
        log_header = "BEGIN RUN %s ====" % time.asctime()
        l.debug(log_header) 

    if LOG == True:
        l.info("trade_size_usd = %d", trade_size_usd)
        l.info("signal_window_time = %d", signal_window_time)
        l.info("min_window_signals = %d", min_window_signals)
        l.info("min_profit_prct = %d", min_profit_prct)
        l.info("usd_transaction_cost = %d", usd_transaction_cost)
        l.info("min_time_between_trades = %d", min_time_between_trades)
        l.info("carry_position = %d", carry_position)
        if max_position is None:
            l.info("max_position = +/- inf")
        else:
            l.info("max_position = %d", max_position)
        l.info("trade_size_scalar = %d", trade_size_scalar)

    # how much of the payment currency can we buy with 1 dollar? 
    conversion_rate = get_cross_rate(currency_pair[1])
    trade_size = trade_size_usd * conversion_rate 
    transaction_cost = usd_transaction_cost * conversion_rate 
    VOLUME_SCALAR = 1000000.

    
    BUY = +1
    SELL = -1 

    POS_LIMIT = +2
    MIN_TIME = +3
    MIN_PNL = +4
    CUT = +5
    
    position = 0 
    position_price = 0
    last_trade_time = 0
    
    n = len(ts)
    profits = np.zeros(n)
    position_deltas = np.zeros(n) # trade sizes
    position_running = np.zeros(n) # cumulative position cheaper than position_running[i] = sum(position_deltas[i:i])
    buy_signals_window = []
    sell_signals_window = []
    ignored_signals = np.zeros(n)
    count_ignored = 0
    raw_buy_count = 0   # sanity checks
    raw_sell_count = 0  # sanity checks
    raw_none_count = 0  # sanity checks
    windowed_buy_count = 0
    windowed_sell_count = 0
    m2m_pnl = np.zeros(n)
    level_slippage = 0.00005 # prices get worse by 1/2 pip per level
    cut_price_delta_long = -0.0005 # allow this much absolute price variance from position price before cutting
    cut_price_delta_short = -0.0005
    
    #action_indices = np.nonzero(signal)[0] 
    #for i in action_indices:
    for i in xrange(n):
        curr_offer = offers[i]
        curr_bid = bids[i] 
        curr_offer_vol = offer_vols[i]
        curr_bid_vol = bid_vols[i]

        # TODO sanity check ts for discontinuitities
        t = ts[i] 
        curr_signal = signal[i]
        if curr_signal == BUY: raw_buy_count += 1        
        if curr_signal == SELL: raw_sell_count += 1        
        if curr_signal == 0: raw_none_count += 1

        # if we have a position get the mark to market (m2m) pnl
        if position > 0:
            # TODO should assume we can sweep the book and get uniform amt at each 
            # level with slippage
            # e.g. 1000000 with 250k at each level would take bid at level 0; 
            # then bid - 1*level_slippage at level 2; bid - 2*level_slippage at level 2 etc...
            worst_price_depth = int(position / (curr_bid_vol * VOLUME_SCALAR))
            worst_price_adjust = (worst_price_depth * level_slippage)
            if LOG: l.debug("LONG: position_price(%f) curr_mkt(%f@%f) worst_price_adjust(%f) position(%f) worst_price_depth(%f) curr_bid_vol(%f)", position_price, curr_bid, curr_offer, worst_price_adjust, position, worst_price_depth, curr_bid_vol)
            m2m_pnl[i] = (curr_bid - worst_price_adjust - position_price) * abs(position)
            if LOG: l.debug("LONG M2M pnl: %f", m2m_pnl[i])

            if LOG: l.debug("Checking for curr_bid - position_price: %f < cut_price_delta_long: %f", curr_bid - position_price, cut_price_delta_long)
            if (curr_bid - position_price - worst_price_adjust) < cut_price_delta_long:
                if LOG: l.debug("Cut long pos")
                position = 0
                position_running[i] = 0 
                position_deltas[i] = position
                last_trade_time = t
                profits[i] = -transaction_cost
                ignored_signals[i] = CUT
                continue
            else:
                if LOG: l.debug("NOT CUTTING")

        elif position < 0:
            worst_price_depth = int(position / (curr_offer_vol * VOLUME_SCALAR))
            worst_price_adjust = (worst_price_depth * level_slippage)
            if LOG: l.debug("SHORT: position_price(%f) curr_mkt(%f@%f) worst_price_adjust(%f) position(%f) worst_price_depth(%f) curr_bid_vol(%f)", position_price, curr_bid, curr_offer, worst_price_adjust, position, worst_price_depth, curr_bid_vol)
            m2m_pnl[i] = (position_price - curr_offer - worst_price_adjust) * abs(position)
            if LOG: l.debug("SHORT M2M pnl: %f", m2m_pnl[i])

            if LOG: l.debug("Checking for position_price - curr_offer: %f < cut_price_delta_short: %f", position_price - curr_offer, cut_price_delta_short)
            if (position_price - curr_offer - worst_price_adjust) < cut_price_delta_short:
                if LOG: l.debug("Cut short pos")
                position = 0
                position_running[i] = 0
                position_deltas[i] = position
                last_trade_time = t
                profits[i] = -transaction_cost
                ignored_signals[i] = -CUT
                continue
            else:
                if LOG: l.debug("NOT CUTTING")

        # P(fill) should depend on time between trades
        #if t - last_trade_time  > min_time_between_trades:
        if LOG: l.debug("Current signal: %s", curr_signal)
        if curr_signal != 0:

            trade_size = trade_size_usd * conversion_rate 
            if LOG: l.debug("Reset trade size to: %d trade_size_usd(%d) conversion_rate(%f)", trade_size, trade_size_usd, conversion_rate)

            if curr_signal == BUY: buy_signals_window.append(t)
            elif curr_signal == SELL: sell_signals_window.append(t) 
            # clean old signals from window 
            cutoff = t - signal_window_time 
            # Changed cutoff bounds to (...] - to have [...] bounds change > to >=
            buy_signals_window = [btime for btime in buy_signals_window if btime > cutoff]
            sell_signals_window = [stime for stime in sell_signals_window if stime > cutoff]
                

            # should we buy? 
            nbuys = len(buy_signals_window)
            windowed_buy_count += nbuys
            nsells = len(sell_signals_window)
            windowed_sell_count += nsells

            if LOG: l.debug("buy signals count: %i ", nbuys)
            if LOG: l.debug("sell signals count: %i  ", nsells)

            # debug
            #if num_buy_signals > 0: print "buy_signals_window: ", buy_signals_window, '[', num_buy_signals, ']' 
            #if num_sell_signals > 0: print "sell_signals_window: ", sell_signals_window, '[', num_sell_signals, ']'
                
            
            # TODO can we pass a function to evaluate the window signal? 
            # TODO pass a risk function to determine cutoff risk = f(position, m2mpnl, last_trade_time, max_hold_time)
            # buying 
            if nbuys > min_window_signals and nsells == 0: #and num_buy_signals > num_sell_signals:
                if LOG: l.debug("nbuys > min_window_signals and nsells == 0")
                # size the trade correctly
                if curr_offer_vol < (trade_size/VOLUME_SCALAR)*trade_size_scalar:
                    trade_size = int(round(curr_offer_vol * VOLUME_SCALAR * (1./trade_size_scalar), -3))
                    trade_size = trade_size * conversion_rate 
                    if LOG: l.debug("Setting long trade size to: %d", trade_size)

                if position >= 0: 
                    if max_position is None or (position + trade_size) < max_position:
                        if fill_function is None or fill_function():
                            if LOG: l.debug("TRADE: BUY %d @ %f", trade_size, curr_offer)
                            position_price = (position * position_price + curr_offer * trade_size) / (trade_size + position)
                            position += trade_size
                            position_running[i] = position 
                            position_deltas[i] = trade_size
                            last_trade_time = t
                            profits[i] = -transaction_cost
                            if LOG: l.debug("Long position after trade (%f) @ %f", position, position_price)
                        else:
                            if LOG: l.debug("Missed adding to long pos with buy - no trade")
                    else:
                        if LOG: l.debug("LIMIT LONG - cur pos: %f", position)
                        ignored_signals[i] = POS_LIMIT
                elif position < 0:
                    # Always allow closing a position so no check for max_position
                    profit_prct = (position_price - curr_offer) / position_price
    
                    # Cover short ONLY at profit
                    if min_profit_prct is None or profit_prct >= min_profit_prct:
                        if fill_function is None or fill_function():
                            if LOG: l.debug("TRADE: BUY %d @ %f", trade_size, curr_offer)
                            abs_pos = abs(position)
                            position_price = (abs_pos * position_price + curr_offer * trade_size) / (trade_size + abs_pos) 
                            position += trade_size
                            position_running[i] = position 
                            position_deltas[i] = trade_size
                            last_trade_time = t
                            curr_profit = profit_prct * trade_size
                            profits[i] = curr_profit - transaction_cost
                            if LOG: l.debug("Taking profit cover short pos %f @ %f", position, position_price)
                        else:
                            if LOG: l.debug("Missed covering short pos - no trade")
                    else:
                        if LOG: l.debug("Insufficient profit to cover short pos %f @ %f", position, position_price)
                        ignored_signals[i] = MIN_PNL
            # selling
            elif nsells > min_window_signals and nbuys == 0: #and num_sell_signals > num_buy_signals:
                #size the trade correctly
                if curr_bid_vol < (trade_size/VOLUME_SCALAR)*trade_size_scalar:
                    trade_size = int(round(curr_bid_vol * VOLUME_SCALAR * (1./trade_size_scalar), -3))
                    trade_size = trade_size * conversion_rate 
                    if LOG: l.debug("Setting short trade size to: %f (current bid vol: %f)", trade_size, curr_bid_vol)
                 


                if position <= 0:
                    if max_position is None or (position - trade_size) > -max_position:
                        if fill_function is None or fill_function():
                            if LOG: l.debug("TRADE: SELL %d @ %f", trade_size, curr_bid)
                            abs_pos = abs(position)
                            position_price = (abs_pos * position_price + curr_bid * trade_size) / (trade_size + abs_pos)    
                            position_deltas[i] = -trade_size
                            position -= trade_size
                            position_running[i] = position 
                            last_trade_time = t
                            profits[i] = -transaction_cost
                            if LOG: l.debug("Short position after trade (%f) @ %f", position, position_price)
                        else:
                            if LOG: l.debug("Missed selling short - no trade")
                    else:
                        if LOG: l.debug("LIMIT SHORT - cur pos: %f", position)
                        ignored_signals[i] = -POS_LIMIT
                elif position > 0:
                    # Always allow closing long position so no check for max_position
                    profit_prct = (curr_bid - position_price) / position_price

                    # Sell long position ONLY at profit
                    if min_profit_prct is None or profit_prct >= min_profit_prct:
                        if fill_function is None or fill_function():
                            position_price = (position * position_price + curr_offer * trade_size) / (trade_size + position)
                            position_deltas[i] = -trade_size
                            position -= trade_size
                            position_running[i] = position 
                            last_trade_time = t
                            curr_profit = profit_prct * trade_size
                            profits[i] = curr_profit - transaction_cost
                            if LOG: l.debug("Taking profit selling long pos %d @ %f", position, position_price)
                        else:
                            if LOG: l.debug("Missed selling long pos - no trade")
                    else:
                        if LOG: l.debug("Insufficient profit to sell long pos %d @ %f ", position, position_price)
                        ignored_signals[i] = -MIN_PNL 
        else:
            if curr_signal == BUY:
                ignored_signals[i] = MIN_TIME
            elif curr_signal == SELL:
                ignored_signals[i] = -MIN_TIME

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
    print "Raw buy signal count: ", raw_buy_count, " window count: ", windowed_buy_count
    print "Raw sell signal count:", raw_sell_count," window count: ", windowed_sell_count
    return (usd_profits, position_deltas, position_running, usd_last_position, ignored_signals)


""" 
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
"""
