import numpy as np
import scipy.stats as stats
import logging 
import random
import time
import dataset
import sys
import csv
import os

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

def load_dataset(hdf_file):
    """
    load_dataset(path_to_hdf_file):

    return (d, currency_pair,  ts, bids, offers, bid_vols, offer_vols)
    for use with execute_aggressive 
    """
    d = dataset.Dataset(hdf_file)
    ts = d['t']
    bids = d['bid']
    offers = d['offer']
    bid_vols = d['bid_vol']
    offer_vols = d['offer_vol']
    currency_pair = d.currency_pair
    (h,m,s,ms) = millis_to_hmsms(ts[-1]-ts[0])
    duration_str = "%d hours, %d minutes, %d seconds, %d ms" % (h, m, s, ms)
    print "Loading: ", "[", ts[0], "-", ts[-1], "]", duration_str 
    return (d, currency_pair, ts, bids, offers, bid_vols, offer_vols)


MILLIS_DAY = 24*60*60*1000;
def millis_to_hmsms(millis):
    seconds = millis/1000
    h = seconds / 3600 
    m = (seconds - (h*3600))/60
    ss = (seconds - ((h*3600)+(m*60)))
    ms = millis - ((ss*1000)+(m*60*1000)+(h*3600*1000))
    if h > 24 or m > 60 or ss > 60 or ms > 1000: 
        raise RuntimeError("Invalid time: " , h , ':' , m , ':' , ss , '.' , ms)
    return (h, m, ss, ms)

def get_cross_rate(currency):
    currency = currency.lower()
    if currency in old_usd_rates:
        return old_usd_rates[currency]
    else: 
        raise RuntimeError("Currency unknown: " + currency)

def trade_stats(d, \
                strategy_name, \
                ts, \
                signals, \
                min_profit_prct, \
                signal_window_time, \
                min_window_signals, \
                trade_size_usd, \
                max_position, \
                usd_pnl, \
                pos_deltas, \
                closing_position, \
                closing_pnl, \
                mean_spread, \
                mean_range, \
                ignored_signals, \
                tick_file="",\
                out_path="./"):

    # create logger
    l = logging.getLogger()
    l.setLevel(logging.DEBUG)
    # create file handler and set level to debug
    #fh = logging.FileHandler(out_file, 'w')
    sh = logging.StreamHandler()
    #fh.setLevel(logging.DEBUG)
    sh.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(message)s')
    # add formatter to ch
    #fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    # add ch to logger
    #l.addHandler(fh)
    #logging.getLogger().addHandler(sh)

    currency_pair = d.currency_pair

    non_zero_signals = np.nonzero(signals)
    signal_count = np.count_nonzero(signals)
    buys = [signals[x] == 1 for x in non_zero_signals]
    sells = [signals[x] == -1 for x in non_zero_signals]


    abs_deltas = abs(pos_deltas)
    max_trade_size = max(pos_deltas)
    min_trade_size = min(pos_deltas)
    nz = np.nonzero(abs_deltas)
    nz_abs_deltas = [abs_deltas[x] for x in nz]
    total_turnover = sum(abs_deltas)
    mean_trade_size = np.mean(nz_abs_deltas[0])
    total_pnl = sum(usd_pnl)

    nz = np.nonzero(usd_pnl)
    nz_pnl = [usd_pnl[x] for x in nz]
    ntrades = len(nz_pnl[0])
    if ntrades > 0:
        mean_pnl = np.mean(nz_pnl[0])
        min_trade_pnl = min(nz_pnl[0])
        max_trade_pnl = max(nz_pnl[0])
    else:
        mean_pnl = 0
        min_trade_pnl = 0
        max_trade_pnl = 0
        
    (h,m,s,ms) = millis_to_hmsms(ts[-1]-ts[0])
    duration_str = "%d hours, %d minutes, %d seconds, %d ms" % (h, m, s, ms)
    timeseries_millis = ts[-1] - ts[0]
    
    
    #buy_count = np.count_nonzero(buys)
    #sell_count = np.count_nonzero(sells)
    attrs = [
            'time',
            'tick file', 
            'strategy name', 
            'currency pair', 
            'min profit prct',
            'signal window time',
            'min window signals',
            'trade size (USD)',
            'max position',
            'duration',
            'timseries millis', 
            'number of trades',
            'raw signal count',
            'total pnl (USD)', 
            'worst trade pnl (USD)', 
            'best trade pnl (USD)', 
            'turnover (USD)', 
            'mean trade size',
            'min trade size',
            'max trade size',
            'number buy signals',
            'number sell signals', 
            'mean spread',
            'mean range',
            'cutoff', 
            'limit - long pos',
            'limit - short pos',
            'limit - long time',
            'limit - short time',
            'limit - pnl long miss',
            'limit - pnl short miss',
            'cuts - long',
            'cuts - short'
            ]
    write_header = False  
    if not os.path.isfile(out_path+strategy_name + ".csv"):
          write_header = True  

    f = open(out_path + strategy_name + ".csv", "ab") 
    csv_writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if write_header:
        csv_writer.writerow(attrs)
    csv_writer.writerow([time.asctime(time.gmtime()), \
                    tick_file, \
                    strategy_name, \
                    currency_pair[0]+currency_pair[1], \
                    min_profit_prct, \
                    signal_window_time, \
                    min_window_signals, \
                    trade_size_usd, \
                    max_position, \
                    duration_str, \
                    timeseries_millis, \
                    ntrades, \
                    signal_count, \
                    total_pnl, \
                    min_trade_pnl, \
                    max_trade_pnl, \
                    total_turnover, \
                    mean_trade_size, \
                    min_trade_size, \
                    max_trade_size, \
                    np.count_nonzero(buys), \
                    np.count_nonzero(sells), \
                    mean_spread, \
                    mean_range, \
                    mean_spread + mean_range, \
                    sum(ignored_signals == 2), \
                    sum(ignored_signals == -2), \
                    sum(ignored_signals == 3), \
                    sum(ignored_signals == -3), \
                    sum(ignored_signals == 4), \
                    sum(ignored_signals == -4), \
                    sum(ignored_signals == 5), \
                    sum(ignored_signals == -5)])
    f.close()                  
    
    trade_by_trade = [usd_pnl[x] for x in non_zero_signals]
    l.debug("RUN: \"%s\"", time.asctime(time.gmtime()))
    l.debug("Tick file: \"%s\"", tick_file)
    l.debug("Strategy name: \"%s\"", strategy_name)
    l.debug("Min profit prct: %f", min_profit_prct)
    l.debug("Signal window time: %d", signal_window_time)
    l.debug("Min window signals: %d", min_window_signals)
    l.debug("Trade size (USD): %f", trade_size_usd)
    l.debug("Max position: %d", max_position)
    l.debug("%s %s %d %s %d %s %s", "Timeseries: ", "[", ts[0], "-", ts[-1], "]", duration_str )
    l.debug("Number of trades: %d", ntrades)
    l.debug("Signal count: %d", signal_count)
    l.debug("Total PnL (USD): %d", total_pnl)
    l.debug("Min trade PnL (USD): %d", min_trade_pnl)
    l.debug("Max trade PnL (USD): %d", max_trade_pnl)
    l.debug('Total turnover (USD): %d', total_turnover)
    l.debug('Mean trade size: %f', mean_trade_size)
    l.debug('Min trade size: %f', min_trade_size)
    l.debug('Max trade size: %f', max_trade_size)
    l.debug("Num Buy signals: %d", np.count_nonzero(buys))
    l.debug("Num Sell signals: %d", np.count_nonzero(sells))
    
    l.debug('Mean spread: %f', mean_spread)
    l.debug('Mean range: %f', mean_range)
    l.debug('Cutoffs set at +/-: %f', (mean_spread + mean_range))
    # constants for denoting actions at each t
    #POS_LIMIT = +2
    #MIN_TIME = +3
    #MIN_PNL = +4
    #CUT = +5
    l.debug("%s %s %d %s %d", "Position limit hit:", "long:", sum(ignored_signals == 2), " short: ", sum(ignored_signals == -2) )
    l.debug("%s %s %d %s %d", "Min time between trades:", "long:", sum(ignored_signals == 3), " short: ", sum(ignored_signals == -3) )
    l.debug("%s %s %d %s %d", "Min PnL missed:", "long:", sum(ignored_signals == 4), " short: ", sum(ignored_signals == -4) )
    l.debug("%s %s %d %s %d", "Cuts", "long:", sum(ignored_signals == 5), " short: ", sum(ignored_signals == -5) )
    #return (signal_count, buys, sells, trade_by_trade)

def fill_binomial(n=1, p=0.8):
   return stats.binom.ppf(np.random.rand(), n, p) 

# N.B. sizes are scaled by 1e6 (divide by 1000000)
def size_trade(desired_size, avail_size, trade_size_scalar, VOLUME_SCALAR=1000000):
    if desired_size < avail_size:
        trade_size = desired_size 
    else:
        trade_size = avail_size * (1./trade_size_scalar) 
        #trade_size = int(round(avail_size * (1./trade_size_scalar), -3))
        print "size_trade:", trade_size, avail_size, trade_size_scalar, 1./trade_size_scalar, trade_size*VOLUME_SCALAR
    return trade_size*VOLUME_SCALAR


def TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, side, trade_size, curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR=1000000):
    assert(long_vol >= 0)
    assert(short_vol >= 0)
    assert(long_paid >= 0)
    assert(short_recv >= 0)
    assert(trade_size > 0)

    l = logging.getLogger()
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    #l.addHandler(sh)
    #l.debug("REQUEST TRADE %s : long_vol: %f, short_vol: %f, long_paid: %f, short_recv: %f, trade_size: %f, market [%f %f @ %f %f]", side, long_vol, short_vol, long_paid, short_recv, trade_size, curr_bid_vol, curr_bid, curr_offer, curr_offer_vol)
    #print "REQUEST TRADE %s market [%f %f @ %f %f]" % (side, curr_bid_vol, curr_bid, curr_offer, curr_offer_vol)
    if long_vol != 0: 
        long_avg_price = long_paid / long_vol 
        #print "long_avg_price: ", long_avg_price
    else: long_avg_price = 0.0
    if short_vol != 0: 
        short_avg_price = short_recv / short_vol  
        #print "short_avg_price: ", short_avg_price
    else: short_avg_price = 0.0

    this_trade_pnl = 0

    net_position = long_vol - short_vol

    print "\nREQUEST TRADE %s %f : net_pos: %f, long_avg_price: %f, short_avg_price: %f, long_vol: %f, short_vol: %f, long_paid: %f, short_recv: %f, trade_size: %f, \nmarket [%f %f @ %f %f]\n" % (side, trade_size,net_position, long_avg_price, short_avg_price, long_vol, short_vol, long_paid, short_recv, trade_size, curr_bid_vol, curr_bid, curr_offer, curr_offer_vol)

    if net_position > 0:
        price_delta = curr_bid - long_avg_price  
        if side == "S":
            if trade_size >= net_position:
                assert(net_position > 0)
                this_trade_pnl = net_position * price_delta
            elif trade_size < net_position:
                this_trade_pnl = trade_size * price_delta 
        elif side == "B":
            this_trade_pnl = 0

    elif net_position < 0:
        price_delta = short_avg_price - curr_offer
        if side == "B":
            if trade_size  >= abs(net_position):
                assert(net_position < 0)
                this_trade_pnl = abs(net_position) * price_delta
            elif trade_size < abs(net_position):
                this_trade_pnl = trade_size * price_delta
        elif side == "S":
            this_trade_pnl = 0

    else:
        assert(net_position == 0)
    
    worst_price_adjust = 0.0
    if side == "B":
            if trade_size > (curr_offer_vol * VOLUME_SCALAR): 
                print "WARN: trade_size: %f exceeds available offer_vol: %f" % (trade_size, curr_offer_vol * VOLUME_SCALAR)
                # Adjust price for going through book if needed
                worst_price_adjust = get_worst_price_adjustment(trade_size, curr_offer_vol * VOLUME_SCALAR)
                print "PRICE ADJUST: %f" % (worst_price_adjust)
            long_vol += trade_size
            long_paid += (trade_size * (curr_offer + worst_price_adjust)) 
    elif side == "S":
            if trade_size > (curr_bid_vol * VOLUME_SCALAR):
                print "WARN: trade_size: %f exceeds available bid_vol: %f" % (trade_size, curr_bid_vol * VOLUME_SCALAR)
                # Adjust price for going through book if needed
                worst_price_adjust = get_worst_price_adjustment(trade_size, curr_bid_vol * VOLUME_SCALAR)
                print "PRICE ADJUST: %f" % (worst_price_adjust)
            short_vol += trade_size
            short_recv += (trade_size * (curr_bid - worst_price_adjust))
    else: 
            raise RuntimeError("Invalid side: ", side)


    # We're flat after this trade so reset everything
    #if long_vol == short_vol:
    #        long_vol = short_vol = long_paid = short_recv = 0  
    return (long_vol, short_vol, long_paid, short_recv, this_trade_pnl)

    # LEVEL_SLIPPAGE = Amount of price change per level in book
def get_worst_price_adjustment(trade_size, curr_volume, LEVEL_SLIPPAGE= 0.00005):
    worst_price_depth = abs(int(trade_size / (curr_volume)))
    #print "===============> worst_price_depth:", worst_price_depth
    worst_price_adjust = (worst_price_depth * LEVEL_SLIPPAGE)
    return worst_price_adjust

def MARK_TO_MARKET_PNL(long_vol, short_vol, curr_bid, curr_offer, long_avg_price, short_avg_price):
    if (long_vol - short_vol) == 0:
        return 0.0
    if (long_vol - short_vol) > 0:
        return (curr_bid - long_avg_price) * (long_vol - short_vol)
    elif (long_vol - short_vol) < 0:
        return (short_avg_price - curr_offer)/ abs(long_vol - short_vol)


# trade_size_usd = size of each trade, converted from USD to 1st currency in pair
# signal_window_time = how long in the past to look for signals
# min_window_signals = how many signals must agree in a window before we trade 
# min_profit_prct = how much profit must be made before we act on a signal 
# usd_transaction_cost = cost per trade 
def execute_aggressive(ts, 
        bids, 
        offers, 
        bid_vols, 
        offer_vols, 
        signal, 
        currency_pair,  
        default_trade_size=1000000, 
        signal_window_time=200, 
        min_window_signals=3, 
        min_profit_prct = 0.0008, 
        usd_transaction_cost=13, 
        min_time_between_trades=1000, 
        carry_position = True,
        max_position = None, 
        trade_size_scalar=2,
        fill_function=fill_binomial,
        cut_long = -0.0005,
        cut_short = -0.0005,
        LOG=False, 
        trade_file=None,
        take_profit_pct=None):

    # Per ticket cost is 0 but we figure out trade fees based on turnever at end of run
    transaction_cost = 0
    # Processed data for volumes in the hdf files used for simulation has been scaled by 1e6 
    VOLUME_SCALAR = 1000000.

    
    # constants for denoting actions at each t
    BUY = +1
    SELL = -1 
    POS_LIMIT = +2 # short verion is negative of long
    MIN_TIME = +3
    MIN_PNL = +4
    CUT = +5
    
    net_position = 0.0
    long_vol = 0.0
    short_vol = 0.0
    long_paid = 0.0
    short_recv = 0.0
    long_avg_price = 0.0
    short_avg_price = 0.0
    last_trade_time = 0
    
    n = len(ts)
    pnl = np.zeros(n)
    position_deltas = np.zeros(n) # individual trade sizes
    position_running = np.zeros(n) 
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
    cut_price_delta_long = cut_long # allow this much absolute price deviation against us from position price before cutting
    cut_price_delta_short = cut_short
    log_trades = False
    last_position = 0

    # Trade logging
    if trade_file is not None:
        log_trades = True
        write_header = False
        timestruct = time.gmtime()
        runtime = "%d%d%d-%d%d%d" % (timestruct.tm_year, timestruct.tm_mon, timestruct.tm_mday, timestruct.tm_hour, timestruct.tm_min, timestruct.tm_sec)
        filename = trade_file + '-' + runtime + '.csv'
        if not os.path.isfile(filename):
            write_header = True  

        attrs = [
            'time_offset', 'action', 'index', 'trade_size', 'trade_price', 'pnl', 'long_vol', 'short_vol', 'long_paid', 'short_recv', 'long_avg_price', 'short_avg_price', 'net_position', 'curr_bid', 'curr_offer', 'cut_level', 'current_price_deviation_raw',  'price_adjustment', 'm2m_pnl']

        f = open(filename, "ab") 
        csv_writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            print attrs
            csv_writer.writerow(attrs)
    
    assert(len(ts) == len(signal))
    for i in xrange(n):
        curr_offer = offers[i]
        curr_bid = bids[i] 
        curr_offer_vol = offer_vols[i]
        curr_bid_vol = bid_vols[i]

        # reset the trade size
        trade_size = default_trade_size

        # TODO sanity check ts for discontinuitities
        t = ts[i] 
        curr_signal = signal[i]
        #print "============================> CURR SIGNAL: ", curr_signal
        if curr_signal == BUY: raw_buy_count += 1        
        if curr_signal == SELL: raw_sell_count += 1        
        if curr_signal == 0: raw_none_count += 1

        if long_paid != 0: long_avg_price = long_paid/long_vol
        else: long_avg_price = 0.0
        if short_recv != 0: short_avg_price = short_recv/short_vol
        else: short_avg_price = 0.0

        
        # Do 2 things before we evaluate the next signal or see if we can take a profit
        # 1) see if we should cut the position
        # 2) if we have a position get the mark to market (m2m) pnl

        if (long_vol - short_vol) > 0:
            # Adjust price for going through book if needed
            worst_price_adjust = get_worst_price_adjustment((long_vol - short_vol), curr_bid_vol * VOLUME_SCALAR)
            # Mark to market pnl
            m2m_pnl[i] = MARK_TO_MARKET_PNL(long_vol, short_vol, curr_bid, curr_offer, long_avg_price, short_avg_price)

            # Cut a long position 
            if (curr_bid - long_avg_price - worst_price_adjust) < cut_price_delta_long:
                position_deltas[i] = -(long_vol - short_vol)
                (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                    TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "S", (long_vol - short_vol), curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
                last_trade_time = t
                assert(this_trade_pnl < 0)
                pnl[i] = this_trade_pnl
                assert(long_vol - short_vol == 0)
                position_running[i] = 0 
                ignored_signals[i] = CUT

                if log_trades is True: 
                    row = [t, "CL", i, (long_vol - short_vol), curr_bid - worst_price_adjust, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price,  (long_vol - short_vol),  curr_bid, curr_offer, round(cut_price_delta_long, 5), curr_bid - long_avg_price, worst_price_adjust, m2m_pnl[i]]
                    print row
                    csv_writer.writerow(row)
                continue
            else:
                pass
                #if DEBUG: print("NOT CUTTING")

        elif (long_vol - short_vol) < 0:
            # Adjust price for going through book by level_slippage amount per level we go through
            worst_price_adjust = get_worst_price_adjustment((long_vol - short_vol), curr_offer_vol * VOLUME_SCALAR)
            # Mark to market pnl
            m2m_pnl[i] = MARK_TO_MARKET_PNL(long_vol, short_vol, curr_bid, curr_offer, long_avg_price, short_avg_price)

            # Cut a short position 
            if (short_avg_price - (curr_offer + worst_price_adjust)) < cut_price_delta_short:
                position_deltas[i] = -(long_vol - short_vol)
                (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                    TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "B", abs(long_vol - short_vol), curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
                last_trade_time = t
                assert(this_trade_pnl < 0)
                pnl[i] = this_trade_pnl
                assert(long_vol - short_vol == 0)
                position_running[i] = 0
                ignored_signals[i] = -CUT

                if log_trades is True: 
                    row = [t, "CS", i, abs(long_vol - short_vol), curr_offer + worst_price_adjust, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol), curr_bid, curr_offer, round(cut_price_delta_short, 5), short_avg_price - curr_offer, worst_price_adjust, m2m_pnl[i]]
                    print row
                    csv_writer.writerow(row)
                continue
            else:
                pass
                #if DEBUG: print("NOT CUTTING")

        # We're done cutting and marking - now we do the following:
        # 1) If we have a signal then act on it
        # 2) Else check to see if we can take a profit somehow - if so, close as much of position as possible (trade_only_on_signal controls if this part is evaluated)
        #######################################################

        #if t - last_trade_time  > min_time_between_trades:

        if curr_signal != 0:

            if curr_signal == BUY: buy_signals_window.append(t)
            elif curr_signal == SELL: sell_signals_window.append(t) 

            # Clean old signals from window 
            cutoff = t - signal_window_time 
            # Changed cutoff bounds to (...] - to have [...] bounds change > to >=
            buy_signals_window = [btime for btime in buy_signals_window if btime > cutoff]
            sell_signals_window = [stime for stime in sell_signals_window if stime > cutoff]

            # Compute signal windows
            nbuys = len(buy_signals_window)
            windowed_buy_count += nbuys

            nsells = len(sell_signals_window)
            windowed_sell_count += nsells


            # TODO can we pass a function to evaluate the window signal? 
            # TODO pass a risk function to determine cutoff risk = f(position, m2mpnl, last_trade_time, max_hold_time)
            # BUYING 
            if nbuys >= min_window_signals and nsells == 0: #and num_buy_signals > num_sell_signals:
                # buying with a long position
                if long_vol - short_vol >= 0: 
                    if max_position is None or ((long_vol - short_vol) + trade_size) < max_position:
                        if fill_function is None or fill_function():
                            # size the trade correctly
                            trade_size = size_trade(trade_size/VOLUME_SCALAR, curr_offer_vol, trade_size_scalar, VOLUME_SCALAR)
                            (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                                TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "B", trade_size, curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
                            position_running[i] = long_vol - short_vol 
                            position_deltas[i] = trade_size
                            last_trade_time = t
                            assert(this_trade_pnl == 0)
                            pnl[i] = this_trade_pnl - transaction_cost

                            if log_trades is True: 
                                row = [t, "BL", i, trade_size, curr_offer, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol),  curr_bid, curr_offer,' ',' ', ' ', m2m_pnl[i]]
                                print row
                                csv_writer.writerow(row)

                        else:
                            ignored_signals[i] = NO_FILL
                    else:
                        ignored_signals[i] = POS_LIMIT


                # buying with a short position
                elif long_vol - short_vol < 0:
                    # Always allow closing a position so no check for max_position
                    pnl_prct = (short_avg_price - curr_offer) / short_avg_price
                    # Cover short ONLY at profit
                    if min_profit_prct is None or pnl_prct >= min_profit_prct:
                        assert(pnl_prct > 0 or min_profit_prct is None)
                        if fill_function is None or fill_function():
                            # If our position is profitable sell as much as the market will bear
                            trade_size = size_trade(abs(long_vol - short_vol)/VOLUME_SCALAR, curr_offer_vol, trade_size_scalar, VOLUME_SCALAR)
                            (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                                TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "B", trade_size, curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
                            position_running[i] = (long_vol - short_vol) 
                            position_deltas[i] = trade_size
                            last_trade_time = t
                            assert(this_trade_pnl > 0 or min_profit_prct is None)
                            pnl[i] = this_trade_pnl - transaction_cost

                            if log_trades is True: 
                                row = [t, "BS", i, trade_size, curr_offer, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol),  curr_bid, curr_offer,' ',' ',' ', m2m_pnl[i]]
                                print row
                                csv_writer.writerow(row)

                        else:
                            ignored_signals[i] = NO_FILL
                    else:
                        ignored_signals[i] = MIN_PNL

            # SELLING
            elif nsells >= min_window_signals and nbuys == 0: #and num_sell_signals > num_buy_signals:

                # selling with a short position
                if (long_vol - short_vol) <= 0:
                    if max_position is None or ((long_vol - short_vol) - trade_size) > -max_position:
                        if fill_function is None or fill_function():
                            #size the trade correctly
                            trade_size = size_trade(trade_size/VOLUME_SCALAR, curr_bid_vol, trade_size_scalar, VOLUME_SCALAR)
                            (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                                TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "S", trade_size, curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
                            position_running[i] = (long_vol - short_vol) 
                            position_deltas[i] = -trade_size
                            last_trade_time = t
                            assert(this_trade_pnl == 0)
                            pnl[i]  = this_trade_pnl - transaction_cost
                            if log_trades is True: 
                                row = [t, "SS", i, trade_size, curr_bid, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol),  curr_bid, curr_offer,' ',' ',' ', m2m_pnl[i]]
                                print row
                                csv_writer.writerow(row)
                        else:
                            pass
                    else:
                        ignored_signals[i] = -POS_LIMIT

                # selling with a long position
                elif (long_vol - short_vol) > 0:
                    # Always allow closing long position so no check for max_position
                    pnl_prct = (curr_bid - long_avg_price) / long_avg_price
                    # Sell long position ONLY at profit
                    if min_profit_prct is None or pnl_prct >= min_profit_prct:
                        assert(pnl_prct > 0 or min_profit_prct is None)
                        if fill_function is None or fill_function():
                            # If our position is profitable sell as much as the market will bear
                            assert((long_vol - short_vol) > 0)
                            trade_size = size_trade(abs((long_vol - short_vol))/VOLUME_SCALAR, curr_bid_vol, trade_size_scalar, VOLUME_SCALAR)
                            (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                                TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "S", trade_size, curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
                            position_running[i] = long_vol - short_vol 
                            position_deltas[i] = -trade_size
                            last_trade_time = t
                            assert(this_trade_pnl > 0 or min_profit_prct is None)
                            pnl[i] = this_trade_pnl - transaction_cost
                            if log_trades is True: 
                                row = [t, "SL", i, trade_size, curr_bid, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol),  curr_bid, curr_offer,' ',' ',' ', m2m_pnl[i]]
                                print row
                                csv_writer.writerow(row)
                        else:
                            ignored_signals[i] = -NO_FILL
                    else:
                        ignored_signals[i] = -MIN_PNL 

        # Check for profit taking without signal 
        elif take_profit_pct is not None : 
            if (long_vol - short_vol) < 0:
                pnl_prct = (short_avg_price - curr_offer) / short_avg_price
                if pnl_prct >= take_profit_pct:
                    assert(pnl_prct > 0)
                    if fill_function is None or fill_function():
                        # If our position is profitable dispose as much as the market will bear
                        trade_size = size_trade(abs((long_vol - short_vol))/VOLUME_SCALAR, curr_offer_vol, trade_size_scalar, VOLUME_SCALAR)
                        (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                            TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "B", trade_size, curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
                        assert(short_avg_price > curr_offer)
                        assert(this_trade_pnl > 0)
                        pnl[i] = this_trade_pnl - transaction_cost
                        position_running[i] = long_vol - short_vol 
                        position_deltas[i] = trade_size
                        last_trade_time = t

                        # Logging
                        if log_trades is True: 
                            row = [t, "BP", i, trade_size, curr_offer, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol),  curr_bid, curr_offer,' ',' ',' ', m2m_pnl[i]]
                            print row
                            csv_writer.writerow(row)
                    else:
                        ignored_signals[i] = NO_FILL

            elif (long_vol - short_vol) > 0:
                pnl_prct = (curr_bid - long_avg_price) / long_avg_price
                if pnl_prct >= take_profit_pct:
                    assert(pnl_prct > 0)
                    if fill_function is None or fill_function():
                        # If our position is profitable sell as much as the market will bear
                        trade_size = size_trade(abs((long_vol - short_vol))/VOLUME_SCALAR, curr_bid_vol, trade_size_scalar, VOLUME_SCALAR)
                        (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                            TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "S", trade_size, curr_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
                        assert(long_avg_price < curr_bid)
                        assert(this_trade_pnl > 0)
                        pnl[i] = this_trade_pnl - transaction_cost
                        position_running[i] = long_vol - short_vol 
                        position_deltas[i] = -trade_size
                        last_trade_time = t

                        # Logging
                        if log_trades is True: 
                            row = [t, "SP", i, trade_size, curr_bid, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol),  curr_bid, curr_offer,' ',' ',' ', m2m_pnl[i]]
                            print row
                            csv_writer.writerow(row)
                    else:
                        ignored_signals[i] = -NO_FILL
            
        else:
            if curr_signal == BUY:
                ignored_signals[i] = MIN_TIME
            elif curr_signal == SELL:
                ignored_signals[i] = -MIN_TIME

    closing_pnl = 0
    closing_position = (long_vol - short_vol)
    m2m_pnl[i] = MARK_TO_MARKET_PNL(long_vol, short_vol, curr_bid, curr_offer, long_avg_price, short_avg_price)
    
    #print "NET POSITION============================>", (long_vol - short_vol)
    if not carry_position: 
        if (long_vol - short_vol) > 0:
            last_bid = np.mean(bids[-100:-5])
            (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "S", closing_position, last_bid, curr_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
            profit = this_trade_pnl 
            pnl[-1] += this_trade_pnl 
            closing_pnl = this_trade_pnl
            position_deltas[-1] += -(long_vol - short_vol) 
            position_running[-1] = (long_vol - short_vol)
            # Logging
            if log_trades is True: 
                row = [t, "SO", i, trade_size, last_bid, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol),  last_bid, curr_offer,' ',' ',' ', m2m_pnl[i]]
                
                print row
                csv_writer.writerow(row)

        elif (long_vol - short_vol) < 0:
            last_offer = np.mean(offers[-100:-5])
            (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = \
                TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "B", abs(closing_position), curr_bid, last_offer, curr_bid_vol, curr_offer_vol, VOLUME_SCALAR)
            profit = this_trade_pnl 
            pnl[-1] += this_trade_pnl 
            closing_pnl = this_trade_pnl
            position_deltas[-1] += -(long_vol - short_vol) 
            position_running[-1] = (long_vol - short_vol)
            # Logging
            if log_trades is True: 
                row = [t, "BO", i, trade_size, last_offer, pnl[i], long_vol, short_vol, long_paid, short_recv, long_avg_price, short_avg_price, (long_vol - short_vol),  curr_bid, last_offer,' ',' ',' ', m2m_pnl[i]]
                print row
                csv_writer.writerow(row)
                
    print "Raw buy signal count: ", raw_buy_count, " window count: ", windowed_buy_count
    print "Raw sell signal count:", raw_sell_count," window count: ", windowed_sell_count
    return (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl)


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
